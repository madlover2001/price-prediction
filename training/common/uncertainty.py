from __future__ import annotations

import numpy as np
import pandas as pd

from training.common.config import BOOTSTRAP_BLOCK_SIZE_MONTHS
from training.common.evaluation import get_common_window_predictions

# Responde a la observacion del docente de que "ventajas minimas de RMSE se interpretan
# como decisivas sin estimar incertidumbre" (ver docs/correcciones_docente.md, punto 8).
#
# Se usa block bootstrap por provincia (no remuestreo de filas i.i.d.): los errores de un
# modelo en meses consecutivos de una misma provincia estan autocorrelacionados (si falla
# en marzo, probablemente tambien falle en abril); remuestrear filas sueltas subestima la
# varianza real de la diferencia de RMSE. Cada replica remuestrea bloques moviles de
# BOOTSTRAP_BLOCK_SIZE_MONTHS meses dentro de cada provincia (nunca mezclando bloques entre
# provincias distintas) hasta cubrir la longitud original de esa provincia.
#
# Tambien se corrige por comparaciones multiples: se reportan 9 comparaciones pareadas
# (3 pares de modelos x 3 productos) con IC al 95% cada una; sin correccion, la
# probabilidad de al menos un falso positivo entre las 9 sube considerablemente. Se aplica
# Holm-Bonferroni (step-down) sobre el conjunto completo de las 9 comparaciones.

N_BOOTSTRAP = 2000
CONFIDENCE = 0.95
DEFAULT_SEED = 42


def _block_bootstrap_indices(group_lengths: list[int], block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Indices (sobre el array concatenado por provincia) para una replica de block
    bootstrap: dentro del segmento contiguo de cada provincia, remuestrea bloques moviles
    de `block_size` filas consecutivas (con reemplazo, posicion de inicio aleatoria) hasta
    cubrir la longitud original de esa provincia. Los bloques nunca cruzan de una
    provincia a otra, porque son series temporales independientes."""
    segments = []
    offset = 0
    for length in group_lengths:
        if length <= block_size:
            # Muy pocas filas para formar un bloque mas corto que el propio segmento:
            # se remuestrea el segmento completo como bloque unico.
            segments.append(rng.integers(offset, offset + length, length))
        else:
            max_start = length - block_size
            collected = 0
            blocks = []
            while collected < length:
                start = offset + int(rng.integers(0, max_start + 1))
                blocks.append(np.arange(start, start + block_size))
                collected += block_size
            segments.append(np.concatenate(blocks)[:length])
        offset += length
    return np.concatenate(segments)


def _group_lengths(provincia: np.ndarray) -> list[int]:
    """Longitudes de los segmentos contiguos de `provincia`. Asume que las filas ya vienen
    agrupadas por provincia (get_common_window_predictions ordena por KEY_COLUMNS, que
    incluye 'provincia'), por lo que basta contar corridas consecutivas de igual valor."""
    lengths = []
    current = None
    count = 0
    for value in provincia:
        if value != current:
            if count:
                lengths.append(count)
            current = value
            count = 1
        else:
            count += 1
    if count:
        lengths.append(count)
    return lengths


def bootstrap_rmse(
    y_true,
    y_pred,
    provincia,
    n_boot: int = N_BOOTSTRAP,
    ci: float = CONFIDENCE,
    block_size: int = BOOTSTRAP_BLOCK_SIZE_MONTHS,
    seed: int = DEFAULT_SEED,
):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    group_lengths = _group_lengths(np.asarray(provincia))

    point_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    boot_rmse = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_bootstrap_indices(group_lengths, block_size, rng)
        boot_rmse[i] = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))

    # Bootstrap "basico"/pivotal: el intervalo se refleja alrededor del RMSE puntual real
    # (calculado sobre los datos sin remuestrear), no se centra en el promedio de las
    # replicas bootstrap. RMSE es un funcional concavo (raiz de un promedio); al usar
    # bloques mas largos el numero de bloques independientes por replica baja, la
    # varianza entre replicas sube, y por a Jensen el promedio de las replicas bootstrap
    # queda sesgado hacia abajo frente al valor real -- se verifico numericamente al
    # migrar de remuestreo i.i.d. a block bootstrap (ver docs/correcciones_docente.md).
    pct_lower, pct_upper = np.percentile(boot_rmse, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    lower = max(0.0, 2 * point_rmse - pct_upper)
    upper = 2 * point_rmse - pct_lower
    return point_rmse, float(lower), float(upper)


def bootstrap_rmse_delta(
    y_true,
    y_pred_a,
    y_pred_b,
    provincia,
    n_boot: int = N_BOOTSTRAP,
    ci: float = CONFIDENCE,
    block_size: int = BOOTSTRAP_BLOCK_SIZE_MONTHS,
    seed: int = DEFAULT_SEED,
):
    """Block bootstrap pareado: en cada replica se remuestrean los mismos bloques (misma
    provincia, mismos meses) para los dos modelos a la vez, preservando la correlacion
    entre sus errores. delta = RMSE(a) - RMSE(b); si el intervalo no contiene 0, la
    diferencia se considera significativa al nivel `ci`. Tambien devuelve un p-valor
    bootstrap de dos colas (proporcion de replicas al otro lado de 0, duplicada), para
    poder aplicar correccion por comparaciones multiples fuera de esta funcion."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred_a, dtype=float)
    y_pred_b = np.asarray(y_pred_b, dtype=float)
    group_lengths = _group_lengths(np.asarray(provincia))

    point_rmse_a = np.sqrt(np.mean((y_true - y_pred_a) ** 2))
    point_rmse_b = np.sqrt(np.mean((y_true - y_pred_b) ** 2))
    point_delta = float(point_rmse_a - point_rmse_b)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_bootstrap_indices(group_lengths, block_size, rng)
        rmse_a = np.sqrt(np.mean((y_true[idx] - y_pred_a[idx]) ** 2))
        rmse_b = np.sqrt(np.mean((y_true[idx] - y_pred_b[idx]) ** 2))
        deltas[i] = rmse_a - rmse_b

    # Mismo ajuste pivotal que en bootstrap_rmse, aplicado a la diferencia de RMSE.
    pct_lower, pct_upper = np.percentile(deltas, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    lower = 2 * point_delta - pct_upper
    upper = 2 * point_delta - pct_lower
    significant = bool(not (lower <= 0 <= upper))
    # p-valor consistente con el mismo ajuste pivotal que el IC (no con la distribucion
    # cruda de replicas, que puede estar sesgada frente al punto real -- ver
    # bootstrap_rmse). Se calcula sobre la distribucion reflejada `2*point_delta - deltas`,
    # que es exactamente la distribucion cuyos percentiles 2.5/97.5 producen `lower`/
    # `upper` arriba: usar la misma distribucion para el IC y el p-valor evita que ambos
    # se contradigan (p.ej. IC que excluye 0 pero p-valor > 0.05, o viceversa).
    reflected = 2 * point_delta - deltas
    p_value = float(min(1.0, 2 * min(np.mean(reflected <= 0), np.mean(reflected >= 0))))
    return point_delta, float(lower), float(upper), significant, p_value


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Correccion de Holm (step-down) sobre un conjunto de p-valores. Mas potente que
    Bonferroni simple pero igual de conservadora en el control del error familiar."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    significant = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            significant[idx] = True
        else:
            break  # step-down: en cuanto uno falla, los siguientes (p mayor) tambien fallan
    return significant


def run_rmse_uncertainty() -> None:
    from training.common.registry import MODEL_RESULTS_DIR, PRODUCTS, write_json

    ci_rows = []
    pairwise_rows = []

    for product_id, product in PRODUCTS.items():
        matched = get_common_window_predictions(product_id)
        if len(matched) < 2:
            continue

        model_names = sorted(matched)
        y_true = matched[model_names[0]]["y_true"].to_numpy()
        provincia = matched[model_names[0]]["provincia"].to_numpy()
        n = len(y_true)

        for model_name in model_names:
            mean_rmse, lower, upper = bootstrap_rmse(y_true, matched[model_name]["y_pred"].to_numpy(), provincia)
            ci_rows.append(
                {
                    "product_id": product_id,
                    "product_name": product.display_name,
                    "model_name": model_name,
                    "n": n,
                    "rmse_bootstrap_mean": mean_rmse,
                    "rmse_ci_lower": lower,
                    "rmse_ci_upper": upper,
                    "confidence": CONFIDENCE,
                    "block_size_months": BOOTSTRAP_BLOCK_SIZE_MONTHS,
                }
            )

        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1 :]:
                delta_mean, lower, upper, significant, p_value = bootstrap_rmse_delta(
                    y_true,
                    matched[model_a]["y_pred"].to_numpy(),
                    matched[model_b]["y_pred"].to_numpy(),
                    provincia,
                )
                pairwise_rows.append(
                    {
                        "product_id": product_id,
                        "product_name": product.display_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n": n,
                        "delta_rmse_a_minus_b": delta_mean,
                        "delta_ci_lower": lower,
                        "delta_ci_upper": upper,
                        "significant_95": significant,
                        "p_value_bootstrap": p_value,
                    }
                )

    # Correccion de Holm-Bonferroni sobre el conjunto completo de comparaciones pareadas
    # (las 9: 3 pares de modelos x 3 productos), no por producto por separado.
    if pairwise_rows:
        p_values = [row["p_value_bootstrap"] for row in pairwise_rows]
        holm_significant = holm_bonferroni(p_values, alpha=1 - CONFIDENCE)
        for row, significant_holm in zip(pairwise_rows, holm_significant):
            row["significant_holm"] = significant_holm

    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ci_path = MODEL_RESULTS_DIR / "rmse_confidence_intervals.csv"
    pairwise_path = MODEL_RESULTS_DIR / "rmse_pairwise_significance.csv"

    pd.DataFrame(ci_rows).to_csv(ci_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False, encoding="utf-8-sig")


def run_ablation_uncertainty() -> None:
    """Punto C.5 de la revision del companero: el bootstrap de `run_rmse_uncertainty`
    responde "¿XGBoost difiere de LSTM/SARIMAX?", pero la hipotesis pregunta "¿anadir
    exogenas mejora el modelo?". Aqui se contrasta cada modelo contra si mismo, `base` vs
    `full` (mismo holdout, mismas filas -- no hace falta interseccion de 3 modelos), con
    el mismo block bootstrap pareado + Holm-Bonferroni sobre las 9 comparaciones
    (3 productos x 3 modelos)."""
    from training.common.registry import MODEL_NAMES, MODEL_RESULTS_DIR, PRODUCTS, prediction_path

    rows = []
    for product_id, product in PRODUCTS.items():
        for model_name in MODEL_NAMES:
            base_path = prediction_path(product_id, model_name, "base")
            full_path = prediction_path(product_id, model_name, "full")
            if not base_path.exists() or not full_path.exists():
                continue
            base_df = pd.read_csv(base_path, encoding="utf-8-sig")
            full_df = pd.read_csv(full_path, encoding="utf-8-sig")
            if base_df.empty or full_df.empty:
                continue
            base_df["fecha"] = pd.to_datetime(base_df["fecha"])
            full_df["fecha"] = pd.to_datetime(full_df["fecha"])

            # base y full comparten el mismo holdout (mismos cortes, independientes del
            # feature_set), pero LSTM puede elegir un window_size distinto por
            # configuracion (C.8), lo que puede recortar filas iniciales de forma
            # distinta -- se empareja explicitamente por (provincia, fecha) en vez de
            # asumir el mismo orden/longitud.
            merged = base_df[["provincia", "fecha", "y_true", "y_pred"]].merge(
                full_df[["provincia", "fecha", "y_true", "y_pred"]],
                on=["provincia", "fecha"],
                how="inner",
                suffixes=("_base", "_full"),
            )
            if merged.empty:
                continue
            merged = merged.sort_values(["provincia", "fecha"]).reset_index(drop=True)

            y_true = merged["y_true_full"].to_numpy()
            provincia = merged["provincia"].to_numpy()
            delta_mean, lower, upper, significant, p_value = bootstrap_rmse_delta(
                y_true, merged["y_pred_base"].to_numpy(), merged["y_pred_full"].to_numpy(), provincia
            )
            rows.append(
                {
                    "product_id": product_id,
                    "product_name": product.display_name,
                    "model_name": model_name,
                    "n": int(len(merged)),
                    # positivo = full mejora sobre base (RMSE base > RMSE full), mismo
                    # signo que delta_rmse_full_vs_base en ablation_summary.csv.
                    "delta_rmse_base_minus_full": delta_mean,
                    "delta_ci_lower": lower,
                    "delta_ci_upper": upper,
                    "significant_95": significant,
                    "p_value_bootstrap": p_value,
                }
            )

    if rows:
        p_values = [row["p_value_bootstrap"] for row in rows]
        holm_significant = holm_bonferroni(p_values, alpha=1 - CONFIDENCE)
        for row, significant_holm in zip(rows, holm_significant):
            row["significant_holm"] = significant_holm

    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_RESULTS_DIR / "ablation_rmse_significance.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
