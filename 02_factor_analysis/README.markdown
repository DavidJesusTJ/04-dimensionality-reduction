# <center><b>Análisis Factorial</b></center>

---

## <b>¿Qué es?</b>

El **Análisis Factorial** es una técnica de reducción de dimensionalidad utilizada principalmente en psicometría, ciencias sociales y ecuaciones estructurales. Su propósito es identificar un conjunto reducido de **factores latentes** que explican la covariación entre un gran número de variables observadas.

La idea central es que, aunque tengamos muchas variables medidas, estas pueden estar influenciadas por un número menor de factores subyacentes no observados que capturan la esencia de la estructura de los datos.

En otras palabras:

* Busca **factores comunes** que expliquen las correlaciones entre variables.
* Permite **simplificar** la estructura de los datos conservando su información esencial.
* Se diferencia del PCA porque asume que las variables están influenciadas por factores latentes y errores específicos.

Se utiliza principalmente para:

* **Ecuaciones estructurales** y construcción de modelos de medida.
* **Identificación de dimensiones latentes** en encuestas, test psicológicos o de mercado.
* **Reducción de ítems** en cuestionarios y validación de escalas.
* Exploración de la **estructura subyacente** de un conjunto de variables.

Ejemplos de preguntas que ayuda a responder:

* ¿Qué dimensiones psicológicas explican las respuestas de un test?
* ¿Se puede reducir un cuestionario de 50 preguntas a unas pocas dimensiones latentes?
* ¿Qué factores latentes explican la covariación entre variables económicas o sociales?

---

## <b>Formulación Matemática</b>

El **Análisis Factorial** modela la covarianza observada entre $p$ variables mediante un número reducido $m$ de **factores latentes**. A continuación se desarrolla con detalle la formulación, estimación y propiedades matemáticas clave.


### Modelo de factores (modelo común)

Para cada observación $x \in \mathbb{R}^p$ se asume:

$$
x = \mu + \Lambda f + \epsilon
$$

donde:

- $\mu \in \mathbb{R}^p$ es el vector de medias.  
- $\Lambda \in \mathbb{R}^{p \times m}$ es la **matriz de cargas factoriales** (factor loadings). La entrada $\lambda_{i j}$ mide la influencia del factor $j$ sobre la variable $i$.  
- $f \in \mathbb{R}^m$ es el vector de factores latentes (comunes).  
- $\epsilon \in \mathbb{R}^p$ es el vector de errores específicos (únicos) para cada variable.  

Supuestos típicos:

- $E[f] = 0$, $E[\epsilon] = 0$.  
- $E[f f^\top] = \Phi$ (matriz de covarianza de factores; si se asume factores ortogonales entonces $\Phi = I_m$).  
- $E[\epsilon \epsilon^\top] = \Psi$ (matriz diagonal con varianzas únicas $\psi_{ii}$).  
- $E[f \epsilon^\top] = 0$ (factores y errores no correlacionados).

De este modo la covarianza poblacional $\Sigma$ de $x$ se escribe como:

$$
\Sigma = \operatorname{Cov}(x) = \Lambda \Phi \Lambda^\top + \Psi.
$$

Cuando se trabaja con variables estandarizadas (media 0, varianza 1), es habitual usar la matriz de correlaciones $R$ en lugar de $\Sigma$.

### Comunidades y unicidades

- La **comunalidad** de la variable $i$ es la varianza explicada por los factores: $h_i^2 = (\Lambda \Phi \Lambda^\top)_{ii}$.  
- La **unicidad** es $\psi_{ii}$ y cumple: $\operatorname{Var}(x_i) = h_i^2 + \psi_{ii}$ (si variables estandarizadas, $\psi_{ii} = 1 - h_i^2$).

En el caso ortogonal ($\Phi = I$): $h_i^2 = \sum_{j=1}^m \lambda_{ij}^2$.

### Identificabilidad y restricciones

El modelo presenta indeterminación por rotación: si $T$ es una matriz invertible $m\times m$,

$$
\Lambda f = (\Lambda T)(T^{-1} f),
$$

por lo que sin restricciones $\Lambda$ y $\Phi$ no son únicos. Para asegurar identifiabilidad se imponen convenciones, por ejemplo:

- **Factores ortogonales**: fijar $\Phi = I_m$ y aplicar restricciones de rotación (p. ej., estructura triangular con signos).  
- **Restricción de rotación**: fijar $m(m-1)/2$ condiciones (p. ej., poner ceros en una submatriz de $\Lambda$).  

Condición necesaria de identificación (simplificada, ortogonal): el número de elementos libres de la representación debe ser ≤ número de elementos distintos de $\Sigma$:

$$
pm + p - \frac{m(m-1)}{2} \le \frac{p(p+1)}{2}.
$$

(La resta $m(m-1)/2$ corresponde a la ambigüedad rotacional entre factores).

### Estimación: enfoque de máxima verosimilitud (ML)

Sea $S$ la matriz muestral de covarianzas (o correlaciones) y $n$ el tamaño muestral. La log-verosimilitud (hasta constantes) bajo normalidad multivariante es:

$$
\ell(\Sigma) = -\frac{n}{2}\left(\log|\Sigma| + \operatorname{tr}(S\Sigma^{-1})\right) + \text{const.}
$$

Sustituyendo $\Sigma(\Lambda,\Psi)=\Lambda\Phi\Lambda^\top+\Psi$, la estimación por ML busca:

$$
(\hat\Lambda,\hat\Psi,\hat\Phi) = \arg\max_{\Lambda,\Psi,\Phi}\; \ell\big(\Lambda\Phi\Lambda^\top+\Psi\big).
$$

No existe solución cerrada: el problema se resuelve mediante métodos numéricos (Newton-Raphson, EM, optimización por gradiente). El criterio de ajuste (discrepancia) minimizado suele escribirse como la divergencia de Kullback-Leibler entre $S$ y $\Sigma(\theta)$.

#### Test de ajuste y estadístico de razón de verosimilitud

El estadístico de razón de verosimilitud para contrastar $H_0$: modelo factorial vs $H_1$: saturado (Σ = S) es

$$
G^2 = -2\left[\ell(\hat\Sigma) - \ell(S)\right],
$$

donde $\ell(S)$ es la verosimilitud saturada. Asintóticamente $G^2 \sim \chi^2_{df}$, con

$$
df = \frac{p(p+1)}{2} - \#\text{(parámetros libres del modelo)}.
$$

(En la práctica se usan correcciones por tamaño muestral y no siempre se cumple la aproximación exacta).

### Algoritmos alternativos y procedimientos prácticos

#### 1. Método de ejes principales (Principal Axis Factoring / PAF)

- Inicializa las comunalidades $h_i^2$ (por ejemplo con SMC — squared multiple correlations).  
- Forma la matriz reducida $S^* = S - \operatorname{diag}(\psi_i^{(0)})$ donde $\psi_i^{(0)} = 1 - h_i^{2(0)}$.  
- Realiza descomposición en autovalores/autovectores de $S^*$ y toma los $m$ autovectores principales para obtener cargas iniciales.  
- Recalcula comunalidades $h_i^2 = \sum_{j=1}^m \lambda_{ij}^2$ y repite hasta convergencia.

PAF resuelve iterativamente un problema de mínimos cuadrados para reproducir la estructura común.

#### 2. Método de mínimos cuadrados generalizado (GLS) / MINRES

- MINRES (minimum residual) minimiza la norma de residuo $\|S - \Lambda\Lambda^\top - \Psi\|$ bajo distintas métricas.

#### 3. EM para Máxima Verosimilitud

Tratar $f$ como variables latentes y aplicar EM:

- **E-step**: calcular $E[f|x]$ y $E[ff^\top|x]$ para cada observación usando estimadores condicionales (ver fórmulas abajo).  
- **M-step**: actualizar $\Lambda$ y $\Psi$ resolviendo ecuaciones normales tipo regresión múltiple con las expectativas calculadas.

Fórmulas clave usadas en E-step (asumiendo $\mu=0$ para simplificar):

- $\Sigma = \Lambda\Phi\Lambda^\top + \Psi$.  
- $E[f|x] = \Phi\Lambda^\top \Sigma^{-1} x$.  
- $E[ff^\top|x] = \Phi - \Phi\Lambda^\top \Sigma^{-1} \Lambda\Phi + E[f|x]E[f|x]^\top$.

M-step actualiza acumuladores y calcula nuevas $\Lambda$ y $\Psi$ (ver referencias clásicas de Tipping & Bishop, Rubin-Thayer).

### Cálculo de puntuaciones factoriales (factor scores)

Aunque los factores son latentes, se pueden estimar puntuaciones individuales $\hat f$ por distintos métodos:

- **Método de regresión (Thomson)**:
  $$
  \hat f_{\text{reg}} = \Phi \Lambda^\top \Sigma^{-1} (x - \mu).
  $$
  Es el predictor lineal óptimo (en MSE) de $f$ dado $x$.

- **Método de Bartlett (mínimos cuadrados ponderados)**:
  $$
  \hat f_{\text{Bartlett}} = (\Lambda^\top \Psi^{-1} \Lambda)^{-1} \Lambda^\top \Psi^{-1} (x - \mu).
  $$
  Produce estimadores no sesgados bajo ciertos supuestos y con covarianza más pequeña para factores estimados.

- **Score basado en componentes (regresión sobre cargas estandarizadas)**: aproximaciones útiles cuando $\Psi$ es difícil de estimar.

### Rotaciones y estructuras interpretables

Dado que la solución ML o PAF es única solo hasta rotación, se aplican **rotaciones** para lograr interpretabilidad:

- **Rotaciones ortogonales** ($Q^\top Q = I$), p. ej. *varimax*, *quartimax*. Nuevas cargas: $\Lambda^* = \Lambda Q$. Varimax maximiza la varianza de los cuadrados de cargas por factor. Objetivo varimax (una de sus formulaciones):

  $$
  \text{maximize}\quad V(Q) = \sum_{j=1}^m\left[\frac{1}{p}\sum_{i=1}^p \lambda_{ij}^{*4} - \left(\frac{1}{p}\sum_{i=1}^p \lambda_{ij}^{*2}\right)^2\right].
  $$

- **Rotaciones oblicuas** (permite correlación entre factores), p. ej. *promax*, *oblimin*. Si $T$ es la transformación (no necesariamente ortogonal), entonces:

  $$
  \Lambda^* = \Lambda T, \qquad \Phi^* = T^{-1} \Phi (T^{-1})^\top.
  $$

  Aquí la matriz de factores $\Phi^*$ ya no es identidad y contiene las correlaciones entre factores.

### Selección del número de factores $m$

Métodos comunes:

- **Criterio de Kaiser**: retener autovalores de la matriz de correlación mayores que 1 (crítico y a veces sobreestima).  
- **Scree plot**: inspección visual del codo en la gráfica de autovalores.  
- **Análisis paralelo (Horn)**: comparar autovalores muestrales con los de datos aleatorios; retener aquellos por encima de la pauta aleatoria.  
- **Velicer's MAP**: minimiza la media de los residuos parciales.  
- **Criterios de información (AIC, BIC)** aplicados a la verosimilitud ML.

### Ajuste del modelo y estadísticos

- **Residuo de covarianza**: $R = S - \hat\Sigma = S - (\hat\Lambda \hat\Phi \hat\Lambda^\top + \hat\Psi)$. Evaluar magnitud y patrón de $R$.  
- **Test χ² (razón de verosimilitud)**: ver arriba (estadístico $G^2$).  
- Índices de ajuste alternativos: RMSEA, CFI, TLI (más comunes en SEM/ecuaciones estructurales).

### Propiedades y condiciones numéricas

- Se requiere $\Psi$ con entradas positivas para que $\Sigma$ sea positiva definida.  
- La convergencia puede verse afectada por baja comunalidad, alta multicolinealidad y mala inicialización.  
- El problema es globalmente no convexo: soluciones convergen a mínimos locales; por eso se recomienda múltiples inicializaciones y validación.

### Resumen matemático (intuición final)

- El Análisis Factorial busca descomponer la covarianza en una parte *común* $\Lambda \Phi \Lambda^\top$ y una *específica* $\Psi$.  
- La estimación ML minimiza la discrepancia entre la matriz muestral $S$ y la matriz reproduccida $\Sigma(\theta)$ resolviendo un problema no lineal mediante métodos iterativos (EM, Newton, optimizadores).  
- Existen alternativas más simples (PAF, MINRES) basadas en autovectores y ajustes iterativos de comunalidades.  
- Las rotaciones (ortogonales u oblicuas) se usan para hacer las cargas interpretables, y la elección de $m$ se apoya en criterios estadísticos y gráficos.

---

## <b>Supuestos del Modelo</b>

Para que el **Análisis Factorial** sea válido y produzca resultados confiables, deben cumplirse ciertos supuestos estadísticos y metodológicos:

- **Relaciones lineales**  
  > Se asume que las variables observadas están relacionadas linealmente con un conjunto de **factores latentes no observados**.  
  > Es decir, los factores explican la covarianza común entre las variables.

- **Normalidad multivariada**  
  > Se suele suponer que las variables siguen una **distribución normal multivariada**.  
  > Aunque el análisis factorial es relativamente robusto a desviaciones moderadas, grandes violaciones pueden sesgar los resultados.

- **Correlación suficiente entre variables**  
  > Las variables deben estar **moderada o fuertemente correlacionadas** entre sí.  
  > Si las correlaciones son muy bajas, no tiene sentido buscar factores comunes.  
  > Se recomienda verificar con índices como el **KMO (Kaiser-Meyer-Olkin)** o el **test de esfericidad de Bartlett**.

- **Número adecuado de factores**  
  > El número de factores debe ser **menor que el número de variables** y elegido cuidadosamente (criterio de Kaiser, scree plot, paralel analysis).  
  > Muy pocos factores → pérdida de información; demasiados factores → sobreajuste y pérdida de interpretabilidad.

- **Independencia de los errores específicos**  
  > Se asume que las **varianzas únicas (errores)** no están correlacionadas entre sí ni con los factores comunes.  
  > Esto garantiza que la covarianza entre variables esté explicada solo por los factores.

- **Tamaño de muestra suficiente**  
  > El análisis factorial requiere un **número de observaciones relativamente grande**.  
  > Una regla común es al menos **5 a 10 observaciones por variable**, o un mínimo de 100-200 casos.

- **Escalamiento de variables**  
  > Dado que el análisis factorial se basa en la matriz de correlaciones/covarianzas, es recomendable **estandarizar las variables** (media = 0, varianza = 1) si están en escalas muy diferentes.

- **Adecuación del modelo al objetivo**  
  > El análisis factorial es apropiado cuando:  
  > - Se busca descubrir **factores latentes** que explican la variabilidad común.  
  > - Se desea **reducir dimensionalidad** manteniendo la estructura de correlaciones.  
  > - Se pretende construir modelos de **ecuaciones estructurales** (SEM).  
  > No es adecuado para problemas de predicción directa sin interés en la estructura latente.

---

## <b>Interpretación</b>

Una vez realizado el **Análisis Factorial**, la clave está en **interpretar los factores latentes y las métricas asociadas**.  
A continuación, se explican los principales elementos de interpretación:

### Cargas factoriales

Las **cargas factoriales** representan la **correlación** entre cada variable observada y cada factor latente.  
Se interpretan de manera similar a los coeficientes de regresión estandarizados.

- Valores cercanos a **1 o -1** → fuerte asociación con el factor.  
- Valores cercanos a **0** → poca relación con el factor.  
- Sirven para **nombrar o etiquetar los factores** según las variables que tienen cargas más altas.

> Ejemplo: Si varias variables sobre “satisfacción laboral” cargan fuertemente en el mismo factor, se puede interpretar como un **factor de satisfacción**.

### Comunalidades

La **comunalidad** ($h_i^2$) de una variable mide la proporción de su varianza explicada por los factores comunes:

$$
h_i^2 = \sum_{j=1}^{m} \lambda_{ij}^2
$$

donde $\lambda_{ij}$ es la carga de la variable $i$ en el factor $j$.

- $h_i^2$ cercano a **1** → la variable está bien explicada por los factores.  
- $h_i^2$ bajo → la variable tiene mucha varianza única o error.

### Varianza única (unicidad)

La unicidad ($u_i^2$) mide la parte de la varianza de una variable **no explicada** por los factores:

$$
u_i^2 = 1 - h_i^2
$$

- Valores altos indican que la variable no se ajusta bien al modelo factorial.  
- Se prefiere que las unicidades sean bajas para asegurar que las variables contribuyen a los factores comunes.

### Valores propios (Eigenvalues)

Los **valores propios** de la matriz de correlaciones indican cuánta varianza total explica cada factor.  

- Factores con eigenvalue > 1 (criterio de **Kaiser**) suelen considerarse significativos.  
- El **Scree Plot** ayuda a identificar el “codo” donde la ganancia de varianza explicada disminuye.

> Cuanto mayor el eigenvalue de un factor, más importante es en la explicación del conjunto de datos.

### Varianza total explicada

La suma de los eigenvalues de los factores retenidos indica el porcentaje de varianza total de las variables observadas que está explicado por los factores.

- Un **% explicado alto** indica que el modelo factorial representa bien la estructura latente.  
- En ciencias sociales, un 50-60% puede ser aceptable; en áreas más técnicas, se espera un porcentaje más alto.

### Rotación de factores

La rotación (varimax, oblimin, promax, etc.) se aplica para lograr una **estructura más interpretable**:

- **Rotación ortogonal (Varimax):** mantiene los factores independientes.  
- **Rotación oblicua (Oblimin/Promax):** permite correlaciones entre factores.  

El objetivo es que cada variable cargue fuertemente en un solo factor, facilitando su interpretación.

### En resumen:

| Elemento              | Qué representa                                                     |
|-----------------------|---------------------------------------------------------------------|
| Cargas factoriales    | Correlación de cada variable con los factores latentes              |
| Comunalidades ($h^2$) | Varianza de la variable explicada por los factores                  |
| Unicidad ($u^2$)      | Varianza no explicada (error específico de la variable)             |
| Eigenvalues           | Importancia de cada factor en la explicación de la varianza total   |
| Varianza explicada    | % de la variabilidad total representada por los factores            |
| Rotación              | Técnica para mejorar la interpretabilidad de los factores           |

---

## <b>Implementación en `factor_analyzer`</b>

```python
from factor_analyzer import FactorAnalyzer

model = FactorAnalyzer(
    n_factors=3,
    rotation='promax',
    method='minres',
    use_smc=True,
    is_corr_matrix=False,
    bounds=(0.005, 1),
    impute='median',
    svd_method='randomized',
    rotation_kwargs=None
)

model.fit(X_train)
X_reduced = model.transform(X_test)
```

---

## <b>Parámetros Cruciales</b>

A continuación, se explican los hiperparámetros más importantes que afectan directamente la calidad y la interpretación del **Análisis Factorial** con `FactorAnalyzer`.

### n_factors — Número de factores

Define cuántos **factores latentes** se extraerán del conjunto de datos.

- **Valor bajo** → puede dejar fuera información importante (**underfactoring**).  
- **Valor alto** → puede generar factores poco interpretables y captar ruido (**overfactoring**).

> Es el parámetro central que determina la dimensionalidad reducida y la capacidad de representar las correlaciones.

### rotation — Método de rotación

Controla cómo se ajustan los ejes de los factores para mejorar la interpretabilidad.

- **`varimax`** → maximiza la varianza de las cargas al cuadrado → factores más simples, más interpretables.  
- **`promax`** → permite correlación entre factores → útil cuando se asume que los constructos están relacionados.  
- **`oblimin`** → similar a `promax`, también admite factores correlacionados.  

> La rotación **no cambia el ajuste global**, pero modifica las **cargas factoriales**, haciendo los factores más fáciles de interpretar.

### method — Método de estimación

Indica cómo se extraen los factores iniciales a partir de la matriz de correlaciones.

- **`minres` (Minimum Residual)** → más usado, busca minimizar residuos entre matriz observada y reproducida.  
- **`ml` (Maximum Likelihood)** → basado en verosimilitud, permite pruebas estadísticas (ej. test de adecuación).  
- **`principal`** → similar a componentes principales, pero con ajuste hacia modelo factorial.  

> La elección del método influye en la precisión del modelo y en los supuestos estadísticos que pueden verificarse.

### use_smc — Varianza extraída inicial

Determina si se usan los **comunalidades iniciales** basadas en la **comunalidad máxima (SMC, squared multiple correlations)**.

- **`True`** → más realista, usa correlación múltiple al cuadrado.  
- **`False`** → asigna valores iniciales iguales, menos preciso.  

> Impacta en la calidad inicial de la estimación y la estabilidad de los resultados.

### rotation_kwargs — Parámetros adicionales de rotación

Permite ajustar configuraciones específicas según el tipo de rotación.

- Ejemplo: en `oblimin`, se puede ajustar el parámetro $\gamma$ para controlar el grado de correlación entre factores.  
- Ejemplo: en `promax`, se ajusta el parámetro de potencia que controla cuán oblicua será la rotación.

> Es crucial si se necesita un mayor control sobre la correlación permitida entre factores.

**Resumen gráfico mental:**

| Parámetro       | Impacta en...                          | Ajustar cuando...                                      |
|-----------------|-----------------------------------------|--------------------------------------------------------|
| `n_factors`     | Cantidad de dimensiones latentes        | Se sospecha under/overfactoring                        |
| `rotation`      | Interpretabilidad de las cargas         | Se requiere factores más simples o correlacionados      |
| `method`        | Forma de extracción inicial             | Se quiere precisión estadística o rapidez              |
| `use_smc`       | Estabilidad en comunalidades iniciales  | El modelo no converge o arroja soluciones inestables   |
| `rotation_kwargs` | Control fino de la rotación           | Se necesita ajustar la correlación entre factores      |

---

## <b>Cálculo Interno del Modelo</b>

Cuando llamas al método `.fit()` de **FactorAnalyzer**, se inicia un procedimiento matemático para **extraer factores latentes** a partir de la matriz de correlaciones o covarianzas de los datos.


### ¿Qué significa "entrenar" el modelo?

Entrenar el modelo significa encontrar dos cosas principales:

1. Un conjunto de **factores latentes** ($F$) que explican la covariación entre las variables observadas.  
2. Unas **cargas factoriales** ($\Lambda$) que indican cuánto contribuye cada variable a cada factor.

El modelo factorial se expresa como:

$$
X = \Lambda F + \epsilon
$$

donde:
- $X$ = variables observadas.  
- $\Lambda$ = matriz de cargas factoriales (qué tan fuerte se asocia cada variable con cada factor).  
- $F$ = factores latentes.  
- $\epsilon$ = errores únicos (varianza no explicada por los factores).

### ¿Qué función se minimiza?

El objetivo es que la **matriz de correlaciones observada ($R$)** se aproxime a la **matriz reproducida por el modelo factorial**:

$$
R \approx \Lambda \Lambda^T + \Psi
$$

donde:
- $\Lambda \Lambda^T$ = varianza común explicada por los factores.  
- $\Psi$ = matriz diagonal de **unicidades** (varianza específica de cada variable).  

El método busca minimizar la diferencia entre $R$ y $\Lambda \Lambda^T + \Psi$.

### ¿Qué hace internamente `.fit()`?

El cálculo se desarrolla en varios pasos:

1. **Inicialización de comunalidades**  
   - Si `use_smc=True`, se usan los **cuadrados de correlaciones múltiples (SMC)** como estimación inicial de la varianza común de cada variable.

2. **Extracción de factores (según `method`)**  
   - **`minres`**: busca minimizar la suma de residuos $(R - \Lambda \Lambda^T - \Psi)$.  
   - **`ml`**: maximiza la verosimilitud del modelo dado $R$.  
   - **`principal`**: aproxima mediante descomposición en componentes principales.

3. **Estimación de cargas factoriales ($\Lambda$)**  
   - Se resuelve un problema matricial para obtener cargas que mejor expliquen las correlaciones.

4. **Rotación de factores (según `rotation`)**  
   - `varimax` (ortogonal) → simplifica la estructura de las cargas.  
   - `promax`, `oblimin` (oblicuas) → permiten correlación entre factores.  

   > La rotación **no cambia la varianza total explicada**, solo reorganiza las cargas para mejorar la **interpretación**.

5. **Cálculo de unicidades ($\Psi$)**  
   - Se estima la varianza que **no es explicada** por los factores comunes para cada variable.

### ¿Qué devuelve al final?

Después de entrenar el modelo, puedes acceder a:

- `loadings_` → matriz de **cargas factoriales** ($\Lambda$).  
- `get_factor_variance()` → proporción de varianza explicada por cada factor.  
- `get_communalities()` → varianza común explicada por los factores para cada variable.  
- `get_uniquenesses()` → varianza única (no explicada).  
- `rotation_` → tipo de rotación aplicada.  

### En resumen

Entrenar un **FactorAnalyzer** significa resolver un **problema de factorización matricial**:  
buscar una representación de la matriz de correlaciones como:

$$
R \approx \Lambda \Lambda^T + \Psi
$$

donde $\Lambda$ revela los **patrones latentes compartidos** entre variables, y $\Psi$ refleja las partes **específicas o error** de cada una.

---

## <b>Casos de uso</b>

El **Análisis Factorial (FA)** es especialmente útil cuando el interés no está solo en reducir la dimensionalidad, sino en **descubrir constructos latentes** que explican la covariación entre variables observadas. A diferencia de PCA o NMF, su enfoque es **modelar la estructura común subyacente** más que la simple varianza total.

### Identificación de factores latentes en ciencias sociales

- Ideal para **psicología, sociología y educación**, donde muchos indicadores miden conceptos no observables (ej. inteligencia, ansiedad, satisfacción).  
- El modelo permite agrupar variables observadas en dimensiones más abstractas.  

> Ejemplo: un cuestionario de personalidad con 50 ítems que se reducen a 5 factores principales.

### Desarrollo y validación de cuestionarios

- Detecta si un conjunto de preguntas realmente mide el mismo constructo.  
- Permite eliminar ítems redundantes o poco informativos.  
- Es la base de la **validación de escalas psicométricas**.  

> Ejemplo: validar que las preguntas de un test de ansiedad efectivamente se agrupan en un solo factor.

### Reducción de dimensionalidad con interpretabilidad

- A diferencia de PCA, los factores representan **causas subyacentes** de las correlaciones, no solo combinaciones matemáticas.  
- Genera dimensiones que tienen **sentido teórico** y no solo técnico.  

> Ejemplo: en marketing, agrupar variables de consumo en factores como “precio”, “calidad percibida” o “fidelidad”.

### Exploración en biología y medicina

- Útil en **neurociencia y genética**, donde muchas variables están correlacionadas.  
- Permite descubrir factores ocultos que explican la covariación entre genes, regiones cerebrales o síntomas.  

> Ejemplo: en un EEG, detectar factores latentes que explican patrones de actividad cerebral.

### Comparación con otros métodos

- **Mejor que PCA** → cuando se busca **interpretación de factores latentes** y no solo reducción matemática.  
- **Preferible a NMF** → cuando los datos no requieren no-negatividad y se busca un modelo más estadístico.  
- **Útil frente a modelos supervisados** → cuando el interés es **exploratorio** y no predictivo.

### ¿Cuándo *NO* usar Análisis Factorial?

- Cuando el objetivo es **predicción directa** (ej. clasificación, regresión).  
- Cuando los datos tienen **pocas variables** y no existe correlación significativa.  
- Si no se cumple la premisa de que las variables comparten **factores comunes**.

### Conclusión

> El **Análisis Factorial** es la opción adecuada cuando se necesita descubrir **constructos latentes, simplificar cuestionarios o explorar estructuras ocultas** en datos altamente correlacionados.  
> Es más provechoso que PCA o técnicas puramente matemáticas cuando la **interpretación teórica de los factores** es crucial.

---

## <b>Profundización matemática</b>

El **Análisis Factorial (FA)** se fundamenta en un modelo probabilístico que busca explicar la covarianza observada entre variables mediante un conjunto reducido de **factores latentes**. En esta sección se detallan las formulaciones matemáticas completas: la función de optimización, estimación de parámetros y los distintos esquemas de rotación.

### 1) Modelo matemático básico

Dadas $p$ variables observadas y $m$ factores latentes:

$$
x = \Lambda f + \epsilon
$$

- $x \in \mathbb{R}^p$: vector de variables observadas.  
- $f \in \mathbb{R}^m$: factores comunes ($m < p$).  
- $\Lambda \in \mathbb{R}^{p \times m}$: matriz de **cargas factoriales**.  
- $\epsilon \in \mathbb{R}^p$: errores específicos (únicos a cada variable).

Hipótesis centrales:
- $\mathbb{E}[f] = 0, \; \text{Cov}(f) = I_m$ (factores no correlacionados con varianza unitaria).  
- $\mathbb{E}[\epsilon] = 0, \; \text{Cov}(\epsilon) = \Psi = \operatorname{diag}(\psi_1, \dots, \psi_p)$ (errores independientes).  
- $f \perp \epsilon$ (independencia entre factores y errores).

La matriz de covarianza poblacional se modela como:

$$
\Sigma = \Lambda \Lambda^\top + \Psi
$$

### 2) Función de optimización

El problema de estimación consiste en ajustar $\Lambda$ y $\Psi$ para que $\Sigma$ se aproxime a la matriz de covarianzas observada $S$:

$$
\min_{\Lambda, \Psi} \; \| S - (\Lambda \Lambda^\top + \Psi) \|_F^2
$$

Dependiendo del **método de extracción** (`method` en `FactorAnalyzer`):

- **Minres (mínimos residuales)**  
  Minimiza los cuadrados de las diferencias entre $S$ y $\Sigma$ en las correlaciones reproducidas:

  $$
  \min_{\Lambda, \Psi} \sum_{i<j} (s_{ij} - \hat{s}_{ij})^2
  $$

- **ML (Máxima verosimilitud)**  
  Basado en la log-verosimilitud multivariante normal:

  $$
  \ell(\Lambda, \Psi) = -\frac{n}{2} \left[ \log|\Sigma| + \operatorname{tr}(S \Sigma^{-1}) \right]
  $$

  Se busca **maximizar** $\ell$ o equivalentemente **minimizar** $-\ell$.

### 3) Estimación iterativa

El ajuste se realiza por métodos numéricos iterativos:

1. **Inicialización** (comúnmente PCA sobre $S$ para estimar cargas iniciales).  
2. **Estimación de $\Lambda$** mediante resolución de ecuaciones propias o gradiente.  
3. **Estimación de $\Psi$** como residuo:  
   $$
   \hat{\psi}_i = s_{ii} - \sum_{j=1}^m \lambda_{ij}^2
   $$
4. Repetición hasta convergencia (criterio sobre cambio de cargas o función objetivo).

### 4) Rotaciones factoriales

Las soluciones iniciales de FA no son únicas: si $Q$ es una matriz ortogonal de tamaño $m \times m$, entonces:

$$
\Lambda^\star = \Lambda Q
$$

genera la misma covarianza $\Lambda^\star {\Lambda^\star}^\top = \Lambda \Lambda^\top$.

Por eso, se aplican **rotaciones** para mejorar la **interpretabilidad**. Las más usadas son:

#### a) Rotaciones ortogonales (mantienen independencia entre factores)

- **Varimax**: maximiza la varianza de las cargas al cuadrado dentro de cada factor, buscando que cada variable cargue fuerte en un factor y débil en los demás.  
  $$
  V = \sum_j \left[ \frac{1}{p} \sum_i \lambda_{ij}^4 - \left( \frac{1}{p} \sum_i \lambda_{ij}^2 \right)^2 \right]
  $$
- **Quartimax**: simplifica las variables (cada variable carga en un solo factor).  
- **Equamax**: balancea varimax y quartimax.  
- **Geomin-ort**: minimiza la media geométrica de las cargas absolutas.

#### b) Rotaciones oblicuas (permiten correlación entre factores)
- **Promax**: primero aplica varimax, luego eleva las cargas a una potencia $k$ para forzar estructura simple, y finalmente rota oblicuamente.  
- **Oblimin**: familia general que controla el grado de correlación permitida entre factores (parámetro $\gamma$).  
- **Oblimax / Quatimin**: variantes que ajustan la función objetivo hacia simplicidad de variables o factores.  
- **Geomin-obl**: versión oblicua de geomin.

### 5) Matriz de comunalidades y unicidades

- **Comunalidad ($h_i^2$)**: proporción de varianza de la variable $i$ explicada por los factores.  
  $$
  h_i^2 = \sum_{j=1}^m \lambda_{ij}^2
  $$
- **Unicidad ($\psi_i$)**: varianza no explicada por los factores.  
  $$
  \psi_i = 1 - h_i^2 \quad \text{(si se estandarizan las variables)}
  $$

### 6) Notas sobre identificabilidad

- El modelo tiene **indeterminación rotacional**: infinitas soluciones matemáticamente equivalentes.  
- La rotación es una convención para encontrar la solución **más interpretable**.  
- Se requieren al menos $p \ge m$ y que $\Lambda$ tenga rango completo para identificar factores.

### 7) Resumen intuitivo

El **Análisis Factorial** busca representar $S \approx \Lambda \Lambda^\top + \Psi$.  
- Estima $\Lambda$ y $\Psi$ resolviendo un problema de mínimos residuales o máxima verosimilitud.  
- La solución inicial es no única → se aplica **rotación** (ortogonal u oblicua).  
- Cada rotación responde a un criterio matemático que favorece la **simplicidad e interpretabilidad** de los factores.

En síntesis: FA combina **descomposición espectral, optimización iterativa y geometría de rotaciones** para descubrir estructuras latentes en datos multivariados.

---

## <b>Recursos para profundizar</b>

**Libros**  
- *Factor Analysis and Related Methods* – R. J. Rummel  
- *Latent Variable Models: An Introduction to Factor, Path, and Structural Equation Analysis* – Jöreskog & Sörbom  
- *Applied Multivariate Statistical Analysis* – Johnson & Wichern  
- *Multivariate Data Analysis* – Hair, Black, Babin & Anderson  

**Cursos**  
- Coursera – *Exploratory Factor Analysis* (Universidad de Ámsterdam)  
- EdX – *Statistical Learning* (Stanford Online)  
- MIT OpenCourseWare – *Multivariate Statistical Analysis*  
- YouTube – *StatQuest with Josh Starmer* (explicaciones claras sobre PCA y FA)  

**Documentación oficial**  
- [factor_analyzer: FactorAnalyzer](https://factor-analyzer.readthedocs.io/en/latest/factor_analyzer.html)

---
