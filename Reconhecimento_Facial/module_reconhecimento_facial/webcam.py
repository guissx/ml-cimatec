"""Webcam: controle de acesso ao vivo e captura de fotos para o cadastro."""

import csv
from datetime import datetime
from pathlib import Path
import sys
import threading

import cv2
from loguru import logger
import numpy as np

from module_reconhecimento_facial.dataset import list_images, write_image
from module_reconhecimento_facial.draw import FONT, annotate
from module_reconhecimento_facial.modeling.enroll import AuthorizedBase, pick_enrollment_face
from module_reconhecimento_facial.modeling.faces import detect_faces
from module_reconhecimento_facial.modeling.recognize import (
    ConfirmationTracker,
    Recognition,
    Status,
    recognize,
)

WINDOW_NAME = "Controle de acesso (q para sair)"
CAPTURE_WINDOW_NAME = "Cadastro (ESPACO captura, q sai)"


class AccessLog:
    """Registro em CSV dos eventos relevantes: acessos liberados e rostos falsos."""

    FIELDS = ("data_hora", "evento", "pessoa", "distancia")

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event: str, person: str | None, distance: float) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()

        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    # Hora local com fuso, para o registro não ficar ambíguo.
                    "data_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "evento": event,
                    "pessoa": person or "",
                    "distancia": round(distance, 4),
                }
            )


class RecognitionWorker(threading.Thread):
    """Roda o reconhecimento fora da thread do vídeo.

    Analisar um quadro leva perto de 1 s na CPU (detecção, embedding e
    anti-spoofing). Na mesma thread da câmera, o vídeo congelaria a cada análise.
    Aqui a câmera continua fluida e as caixas são atualizadas quando cada análise
    termina, sempre com o quadro mais recente.
    """

    def __init__(
        self,
        base: AuthorizedBase,
        threshold: float,
        anti_spoofing: bool,
        tracker: ConfirmationTracker,
        access_log: AccessLog,
    ) -> None:
        super().__init__(daemon=True)
        self.base = base
        self.threshold = threshold
        self.anti_spoofing = anti_spoofing
        self.tracker = tracker
        self.access_log = access_log

        self._lock = threading.Lock()
        # Não usar o nome _stop: a classe Thread já tem um método interno com ele.
        self._stopped = threading.Event()
        self._frame: np.ndarray | None = None
        self._results: list[Recognition] = []
        self._spoof_active = False
        self.error: Exception | None = None

    def submit(self, frame: np.ndarray) -> None:
        """Entrega um quadro para análise, descartando o anterior se ainda não foi analisado."""
        with self._lock:
            self._frame = frame

    @property
    def results(self) -> list[Recognition]:
        with self._lock:
            return self._results

    def stop(self) -> None:
        self._stopped.set()

    def analyze(self, frame: np.ndarray) -> None:
        """Reconhece um quadro, aplica a confirmação e registra os eventos."""
        recognitions = recognize(frame, self.base, self.threshold, self.anti_spoofing)
        recognitions = self.tracker.update(recognitions)

        with self._lock:
            self._results = recognitions

        self._log_events(recognitions)

    def run(self) -> None:
        try:
            while not self._stopped.is_set():
                with self._lock:
                    frame, self._frame = self._frame, None

                if frame is None:
                    self._stopped.wait(0.01)
                    continue

                self.analyze(frame)
        except Exception as error:  # noqa: BLE001 - repassado para a thread principal
            self.error = error

    def _log_events(self, recognitions: list[Recognition]) -> None:
        for person in self.tracker.newly_confirmed:
            rec = next(r for r in recognitions if r.granted and r.closest == person)
            logger.success("ACESSO LIBERADO para {} (distância {:.3f})", person, rec.distance)
            self.access_log.write(Status.GRANTED, person, rec.distance)

        # Registra o começo de cada tentativa com rosto falso, não cada quadro dela.
        spoofs = [rec for rec in recognitions if rec.status is Status.SPOOF]
        if spoofs and not self._spoof_active:
            logger.warning("Rosto falso detectado (foto ou tela). Acesso negado.")
            self.access_log.write(Status.SPOOF, spoofs[0].closest, spoofs[0].distance)
        self._spoof_active = bool(spoofs)


def open_camera(index: int) -> cv2.VideoCapture:
    """Abre a câmera. No Windows, o DirectShow abre bem mais rápido que o backend padrão."""
    if sys.platform == "win32":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir a câmera {index}. Confira se outro programa "
            "(Teams, Zoom, navegador) não está usando a câmera, ou tente --camera 1."
        )

    return capture


def run_webcam(
    base: AuthorizedBase,
    threshold: float,
    anti_spoofing: bool,
    camera: int,
    confirmations: int,
    log_path: Path,
) -> None:
    """Mostra o vídeo com as caixas de acesso até o usuário apertar q, Esc ou fechar a janela."""
    capture = open_camera(camera)
    worker = RecognitionWorker(
        base, threshold, anti_spoofing, ConfirmationTracker(confirmations), AccessLog(log_path)
    )
    worker.start()

    logger.info("Webcam aberta. Aperte 'q' ou Esc na janela do vídeo para sair.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.warning("A câmera parou de enviar imagens.")
                break

            if worker.error is not None:
                raise worker.error

            worker.submit(frame)
            cv2.imshow(WINDOW_NAME, annotate(frame, worker.results))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            # Janela fechada pelo X.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        worker.stop()
        worker.join(timeout=5)
        capture.release()
        cv2.destroyAllWindows()

    logger.info("Registro de acessos em {}", log_path)


def next_photo_path(person_dir: Path) -> Path:
    """Próximo nome livre na pasta da pessoa: 01.jpg, 02.jpg, ... sem sobrescrever nada."""
    taken = {path.name for path in list_images(person_dir)} if person_dir.exists() else set()
    number = 1
    while f"{number:02d}.jpg" in taken:
        number += 1
    return person_dir / f"{number:02d}.jpg"


def capture_photos(person_dir: Path, count: int, camera: int) -> list[Path]:
    """Tira fotos de cadastro pela webcam: ESPAÇO captura, q ou Esc encerra.

    Cada foto passa pela mesma checagem do cadastro (um único rosto, grande o
    suficiente) antes de ser salva, então uma captura ruim é refeita na hora.

    Returns:
        list[Path]: Fotos salvas.

    """
    capture = open_camera(camera)
    saved: list[Path] = []

    logger.info("Olhe para a câmera e aperte ESPAÇO para cada foto ({} no total).", count)
    logger.info("Dica: varie um pouco o ângulo e a expressão entre as fotos.")

    try:
        while len(saved) < count:
            ok, frame = capture.read()
            if not ok:
                logger.warning("A câmera parou de enviar imagens.")
                break

            preview = frame.copy()
            cv2.putText(
                preview,
                f"Foto {len(saved) + 1}/{count}: ESPACO captura, q sai",
                (10, 30),
                FONT,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(CAPTURE_WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(CAPTURE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key != ord(" "):
                continue

            face, problem = pick_enrollment_face(detect_faces(frame))
            if face is None:
                logger.warning("Foto recusada: {}. Tente de novo.", problem)
                continue

            path = next_photo_path(person_dir)
            write_image(frame, path)
            saved.append(path)
            logger.success("Foto {}/{} salva em {}", len(saved), count, path)
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return saved
