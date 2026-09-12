"""
01_entendimiento_negocio.py
EDA y baseline - Proyecto DM, CRISP-DM Fase 1 Entrega 3.
Impacto del precio del diésel en los costos de producción de la cementera SOBOCE (Bolivia).

Requisitos: pandas, matplotlib (para los histogramas/boxplots), scikit-learn (opcional, fases posteriores).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 160)

ARCHIVO = Path("../Data/Impacto_Diesel_Cementera_SOBOCE.csv")

df = pd.read_csv(ARCHIVO)
df["Fecha"] = pd.to_datetime(df["Fecha"])

# KPI explicativo derivado
df["Costo_Diesel_Bs_por_Ton"] = (
    df["Costo_Diesel_Total_Bs"] / df["Volumen_Produccion_Ton"]
)

COLS_KPI = [
    "Precio_Diesel_Aplicado_SOBOCE_Bs_Litro",
    "Costo_Diesel_Bs_por_Ton",
    "Costo_Transporte_Bs_Ton",
    "Costo_Produccion_Bs_Ton",
    "Margen_Bruto_Pct",
]


# ---------------------------------------------------------------------------
# Pregunta 1: ¿Qué datos tengo?
# ---------------------------------------------------------------------------
def pregunta_1_que_datos_tengo(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("1. ¿QUÉ DATOS TENGO?")
    print("=" * 70)
    print("Dimensión:", df.shape)
    print("Rango de fechas:", df["Fecha"].min(), "a", df["Fecha"].max())
    print("\nTipos de variable:")
    print(df.dtypes.value_counts())
    print("\nColumnas categóricas:")
    print(df.select_dtypes(include="object").columns.tolist())


# ---------------------------------------------------------------------------
# Pregunta 2: ¿Hay datos faltantes?
# ---------------------------------------------------------------------------
def pregunta_2_datos_faltantes(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("2. ¿HAY DATOS FALTANTES?")
    print("=" * 70)
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    print("Nulos totales:", int(df.isna().sum().sum()))
    if nulos.empty:
        print("No se encontraron valores nulos en ninguna columna.")
    else:
        print(nulos)
    print("IDs duplicados:", df["ID"].duplicated().sum())


# ---------------------------------------------------------------------------
# Pregunta 3: ¿Existen valores atípicos? (criterio IQR)
# ---------------------------------------------------------------------------
def pregunta_3_valores_atipicos(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("3. ¿EXISTEN VALORES ATÍPICOS?")
    print("=" * 70)
    resultados = []
    for col in columnas:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        atipicos = df[(df[col] < limite_inf) | (df[col] > limite_sup)]
        resultados.append({
            "variable": col,
            "n_atipicos": len(atipicos),
            "pct_atipicos": round(len(atipicos) / len(df) * 100, 2),
            "limite_inf": round(limite_inf, 2),
            "limite_sup": round(limite_sup, 2),
        })
        print(f"{col}: {len(atipicos)} atípicos "
              f"({len(atipicos) / len(df) * 100:.2f}%) | "
              f"rango normal [{limite_inf:.2f}, {limite_sup:.2f}]")

        # Boxplot de evidencia visual
        fig, ax = plt.subplots(figsize=(5, 3))
        df.boxplot(column=col, by="Evento_Regulatorio", ax=ax, rot=45)
        ax.set_title(col)
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(f"boxplot_{col}.png", dpi=120)
        plt.close(fig)

    return pd.DataFrame(resultados)


# ---------------------------------------------------------------------------
# Pregunta 4: ¿Cómo se distribuyen los datos?
# ---------------------------------------------------------------------------
def pregunta_4_distribucion(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("4. ¿CÓMO SE DISTRIBUYEN LOS DATOS?")
    print("=" * 70)
    print("Conteo de registros por régimen regulatorio:")
    print(df["Evento_Regulatorio"].value_counts())

    resumen = (
        df.groupby("Evento_Regulatorio", observed=True)[columnas]
          .agg(["count", "mean", "median", "min", "max"])
          .round(3)
    )
    print("\nResumen por régimen regulatorio:\n", resumen)

    # Histogramas de evidencia visual
    for col in columnas:
        fig, ax = plt.subplots(figsize=(5, 3))
        df[col].hist(bins=30, ax=ax)
        ax.set_title(f"Distribución: {col}")
        plt.tight_layout()
        plt.savefig(f"hist_{col}.png", dpi=120)
        plt.close(fig)

    return resumen


# ---------------------------------------------------------------------------
# Pregunta 5: ¿Qué relaciones existen entre variables?
# ---------------------------------------------------------------------------
def pregunta_5_relaciones(df: pd.DataFrame) -> pd.Series:
    print("\n" + "=" * 70)
    print("5. ¿QUÉ RELACIONES EXISTEN ENTRE VARIABLES?")
    print("=" * 70)
    cols_corr = COLS_KPI + ["Precio_Venta_Cemento_Bs_Bolsa50kg"]
    matriz = df[cols_corr].corr(numeric_only=True)
    corr = matriz["Precio_Diesel_Aplicado_SOBOCE_Bs_Litro"].sort_values(ascending=False)
    print("Correlaciones con precio de diésel:\n", corr.round(4))

    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.matshow(matriz, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(matriz.columns)))
    ax.set_yticks(range(len(matriz.columns)))
    ax.set_xticklabels(matriz.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(matriz.columns, fontsize=6)
    plt.tight_layout()
    plt.savefig("matriz_correlacion.png", dpi=120)
    plt.close(fig)

    return corr


# ---------------------------------------------------------------------------
# Baseline y meta de negocio (KPI del proyecto)
# ---------------------------------------------------------------------------
def calcular_baseline(df: pd.DataFrame, reduccion_meta: float = 0.05) -> tuple[float, float]:
    print("\n" + "=" * 70)
    print("BASELINE Y META DE NEGOCIO")
    print("=" * 70)
    mask_actual = df["Evento_Regulatorio"].str.startswith("DS 5676", na=False)
    actual = df.loc[mask_actual]
    baseline = actual["Costo_Produccion_Bs_Ton"].mean()
    meta = baseline * (1 - reduccion_meta)

    print("Registros régimen actual:", len(actual))
    print(f"Baseline KPI: Bs {baseline:.2f}/ton")
    print(f"Meta de negocio (-{reduccion_meta*100:.0f}%): <= Bs {meta:.2f}/ton")
    return baseline, meta


# ---------------------------------------------------------------------------
# Discusión: ¿qué podría salir mal si entrenamos un modelo sin conocer los datos?
# ---------------------------------------------------------------------------
def discusion_riesgos(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DISCUSIÓN: ¿QUÉ PODRÍA SALIR MAL SIN CONOCER LOS DATOS?")
    print("=" * 70)
    conteo = df["Evento_Regulatorio"].value_counts()
    regimen_actual = next(c for c in conteo.index if c.startswith("DS 5676"))
    n_actual = conteo[regimen_actual]
    print(f"- El régimen actual (DS 5676) representa solo "
          f"{n_actual} de {len(df)} registros ({n_actual / len(df) * 100:.2f}% del total): "
          f"un baseline calculado sobre esta muestra chica es poco robusto.")
    print("- Mezclar regímenes regulatorios sin distinguirlos generaría una relación "
          "promedio que no es válida para ningún régimen en particular.")
    print("- Valores atípicos no tratados pueden inflar o distorsionar el baseline.")
    print("- La alta correlación entre costo de diésel y costo de transporte "
          "(multicolinealidad) puede sesgar la importancia relativa de las variables "
          "en un modelo de regresión.")


if __name__ == "__main__":
    pregunta_1_que_datos_tengo(df)
    pregunta_2_datos_faltantes(df)
    pregunta_3_valores_atipicos(df, COLS_KPI[1:4])  # costos clave
    pregunta_4_distribucion(df, COLS_KPI)
    pregunta_5_relaciones(df)
    calcular_baseline(df)
    discusion_riesgos(df)
