"""Linha de comando do controle de acesso.

python -m module_reconhecimento_facial.main capturar NOME
python -m module_reconhecimento_facial.main cadastrar
python -m module_reconhecimento_facial.main reconhecer
python -m module_reconhecimento_facial.main webcam
python -m module_reconhecimento_facial.main avaliar PASTA_ROTULADA
"""

import csv
import math
from pathlib import Path
from typing import Annotated

from loguru import logger
import typer

from module_reconhecimento_facial.config import (
    ACCESS_LOG_PATH,
    ANTI_SPOOFING,
    AUTHORIZED_DIR,
    EMBEDDINGS_PATH,
    MIN_MARGIN,
    RECOMMENDED_PHOTOS,
    RESULTS_DIR,
    TEST_DIR,
    THRESHOLD,
    WEBCAM_CONFIRMATIONS,
    WEBCAM_INDEX,
)
from module_reconhecimento_facial.dataset import list_images, read_image, write_image
from module_reconhecimento_facial.draw import annotate
from module_reconhecimento_facial.modeling.enroll import AuthorizedBase, load_base
from module_reconhecimento_facial.modeling.recognize import Recognition, recognize

app = typer.Typer(add_completion=False, help="Controle de acesso por reconhecimento facial.")

REPORT_FIELDS = [
    "imagem",
    "rosto",
    "status",
    "pessoa",
    "mais_parecido",
    "distancia",
    "margem",
    "limiar",
    "rosto_real",
]

# Opções repetidas em mais de um comando.
AuthorizedOption = Annotated[Path, typer.Option(help="Pasta com as fotos dos autorizados.")]
ThresholdOption = Annotated[
    float, typer.Option(help="Distância máxima até o template para liberar. Menor = mais rígido.")
]
AntiSpoofingOption = Annotated[
    bool,
    typer.Option(
        "--anti-spoofing/--sem-anti-spoofing",
        help="Nega rostos falsos (foto impressa, foto na tela do celular).",
    ),
]


def _round(value: float) -> float | str:
    return "" if math.isnan(value) or math.isinf(value) else round(value, 4)


def report_rows(image_name: str, recognitions: list[Recognition]) -> list[dict]:
    """Linhas do relatório CSV para uma imagem: uma por rosto, ou uma avisando que não há rosto."""
    if not recognitions:
        return [{"imagem": image_name, "rosto": 0, "status": "sem_rosto"}]

    return [
        {
            "imagem": image_name,
            "rosto": i,
            "status": str(rec.status),
            "pessoa": rec.name or "",
            "mais_parecido": rec.closest or "",
            "distancia": _round(rec.distance),
            "margem": _round(rec.margin),
            "limiar": rec.threshold,
            "rosto_real": "" if rec.face.is_real is None else rec.face.is_real,
        }
        for i, rec in enumerate(recognitions, start=1)
    ]


def process_image(
    path: Path,
    root: Path,
    base: AuthorizedBase,
    threshold: float,
    anti_spoofing: bool,
    output_dir: Path,
) -> list[dict]:
    """Reconhece os rostos de uma foto, grava a versão anotada e devolve as linhas do relatório."""
    image = read_image(path)
    recognitions = recognize(image, base, threshold, anti_spoofing)

    # Mantém a mesma organização de subpastas da pasta de entrada.
    relative = path.relative_to(root) if path != root else Path(path.name)
    write_image(annotate(image, recognitions), output_dir / relative)

    if not recognitions:
        logger.warning("{}: nenhum rosto encontrado", relative)

    for rec in recognitions:
        if rec.granted:
            logger.success(
                "{}: ACESSO LIBERADO para {} (distância {:.3f})", relative, rec.name, rec.distance
            )
        else:
            logger.error(
                "{}: ACESSO NEGADO - {} (mais parecido: {}, distância {:.3f}, limiar {:.3f})",
                relative,
                rec.status,
                rec.closest,
                rec.distance,
                rec.threshold,
            )

    return report_rows(str(relative), recognitions)


@app.command()
def cadastrar(
    autorizados: AuthorizedOption = AUTHORIZED_DIR,
    forcar: Annotated[
        bool, typer.Option(help="Recalcula a base mesmo sem mudanças nas fotos.")
    ] = False,
) -> None:
    """Calcula e salva os embeddings das pessoas autorizadas."""
    base = load_base(autorizados, EMBEDDINGS_PATH, force=forcar)

    for person in base.people:
        logger.info("Autorizado: {} ({} foto(s))", person, base.photo_count(person))


@app.command()
def capturar(
    nome: Annotated[str, typer.Argument(help="Nome da pessoa (vira a pasta em autorizados).")],
    fotos: Annotated[int, typer.Option(help="Quantas fotos tirar.")] = RECOMMENDED_PHOTOS,
    camera: Annotated[int, typer.Option(help="Índice da câmera no OpenCV.")] = WEBCAM_INDEX,
    autorizados: AuthorizedOption = AUTHORIZED_DIR,
) -> None:
    """Tira as fotos de cadastro pela webcam e já cadastra a pessoa."""
    from module_reconhecimento_facial.webcam import capture_photos

    # O nome vira uma pasta: barras ou ".." gravariam fotos fora de autorizados.
    if not nome.strip() or Path(nome).name != nome or nome in {".", ".."}:
        raise typer.BadParameter("Use só o nome, sem barras (ex.: gustavo).")

    saved = capture_photos(autorizados / nome, fotos, camera)
    if not saved:
        logger.warning("Nenhuma foto capturada; cadastro não alterado.")
        return

    cadastrar(autorizados=autorizados, forcar=False)


@app.command()
def reconhecer(
    entrada: Annotated[
        Path, typer.Argument(help="Foto ou pasta de fotos a verificar.")
    ] = TEST_DIR,
    autorizados: AuthorizedOption = AUTHORIZED_DIR,
    saida: Annotated[
        Path, typer.Option(help="Pasta onde as fotos anotadas são salvas.")
    ] = RESULTS_DIR,
    limiar: ThresholdOption = THRESHOLD,
    anti_spoofing: AntiSpoofingOption = ANTI_SPOOFING,
) -> None:
    """Verifica as fotos de teste e salva cada uma com as caixas de acesso."""
    base = load_base(autorizados, EMBEDDINGS_PATH)
    logger.info(
        "Autorizados: {} | limiar: {:.2f} | anti-spoofing: {}",
        ", ".join(base.people),
        limiar,
        "ligado" if anti_spoofing else "DESLIGADO",
    )

    paths = [entrada] if entrada.is_file() else list_images(entrada)
    if not paths:
        raise typer.BadParameter(f"Nenhuma imagem encontrada em {entrada}")

    rows = []
    for path in paths:
        rows.extend(process_image(path, entrada, base, limiar, anti_spoofing, saida))

    saida.mkdir(parents=True, exist_ok=True)
    report_path = saida / "relatorio.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    granted = sum(row["status"] == "liberado" for row in rows)
    denied = sum(row["status"] not in ("liberado", "sem_rosto") for row in rows)
    logger.info(
        "{} imagem(ns): {} rosto(s) liberado(s), {} negado(s). Resultados em {}",
        len(paths),
        granted,
        denied,
        saida,
    )


@app.command()
def webcam(
    camera: Annotated[int, typer.Option(help="Índice da câmera no OpenCV.")] = WEBCAM_INDEX,
    autorizados: AuthorizedOption = AUTHORIZED_DIR,
    limiar: ThresholdOption = THRESHOLD,
    anti_spoofing: AntiSpoofingOption = ANTI_SPOOFING,
    confirmacoes: Annotated[
        int, typer.Option(help="Análises seguidas reconhecendo a pessoa antes de liberar.")
    ] = WEBCAM_CONFIRMATIONS,
) -> None:
    """Controle de acesso ao vivo pela webcam. Aperte "q" ou Esc para sair."""
    from module_reconhecimento_facial.webcam import run_webcam

    base = load_base(autorizados, EMBEDDINGS_PATH)
    logger.info("Autorizados: {}", ", ".join(base.people))

    if not anti_spoofing:
        logger.warning("Anti-spoofing DESLIGADO: uma foto do autorizado pode liberar o acesso.")

    run_webcam(base, limiar, anti_spoofing, camera, confirmacoes, ACCESS_LOG_PATH)


@app.command()
def avaliar(
    pasta: Annotated[Path, typer.Argument(help="Base rotulada: uma subpasta por pessoa.")],
    fotos_cadastro: Annotated[int, typer.Option(help="Fotos de cadastro por pessoa.")] = 3,
    pessoas: Annotated[int, typer.Option(help="Pessoas cadastradas na simulação.")] = 100,
    impostores: Annotated[
        int, typer.Option(help="Pessoas não cadastradas tentando entrar.")
    ] = 300,
    margem: Annotated[float, typer.Option(help="Margem mínima entre 1º e 2º.")] = MIN_MARGIN,
    semente: Annotated[int, typer.Option(help="Semente do sorteio.")] = 42,
) -> None:
    """Mede falsos aceites e falsas rejeições em uma base rotulada, para escolher o limiar."""
    from module_reconhecimento_facial.modeling.evaluate import run_evaluation

    rows = run_evaluation(pasta, fotos_cadastro, pessoas, impostores, semente, margem)

    print(f"\n{'limiar':>7} {'FRR':>7} {'FAR':>7} {'impostores':>11} {'outra pessoa':>13}")
    for row in rows:
        marker = "  <- atual" if math.isclose(row["limiar"], THRESHOLD) else ""
        print(
            f"{row['limiar']:>7.2f} {row['FRR']:>7.1%} {row['FAR']:>7.2%} "
            f"{row['impostores_liberados']:>11} {row['liberados_como_outra_pessoa']:>13}{marker}"
        )

    safe = [row for row in rows if row["FAR"] == 0]
    if safe:
        best = max(safe, key=lambda row: row["limiar"])
        logger.success(
            "Maior limiar sem nenhum falso aceite: {:.2f} (FRR {:.1%})",
            best["limiar"],
            best["FRR"],
        )
    else:
        logger.warning("Nenhum limiar testado ficou sem falsos aceites.")


if __name__ == "__main__":
    app()
