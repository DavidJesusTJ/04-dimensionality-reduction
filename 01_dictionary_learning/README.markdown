# <center><b> DictionaryLearning </b></center>

---

## <b>¿Qué es?</b>

**DictionaryLearning** es una técnica de reducción de dimensionalidad y representación esparsa que busca expresar los datos como una combinación lineal de un número reducido de componentes básicos llamados átomos o diccionario.

La idea central es que, aunque los datos originales tengan muchas dimensiones, estos pueden describirse mediante un diccionario más pequeño y eficiente, donde cada muestra se representa usando pocos elementos de este diccionario.

En otras palabras:

* Construye un diccionario de patrones a partir de los datos.
* Cada observación se expresa como una suma de pocos elementos del diccionario.
* Esto produce una representación más compacta, interpretable y robusta al ruido.

Se utiliza principalmente para:

* Reducción de dimensionalidad (similar a PCA pero con restricciones de esparsidad).
* Extracción de características para clasificación o clustering.
* Procesamiento de imágenes (ej. eliminación de ruido, compresión).

Ejemplos de preguntas que ayuda a responder:

* ¿Cómo representar datos de alta dimensión con pocos patrones significativos?
* ¿Qué componentes básicos describen mejor un conjunto de imágenes o señales?
* ¿Cómo eliminar ruido conservando las características principales de los datos?

---

## <b>Formulación Matemática</b>

El modelo de **DictionaryLearning** busca representar un conjunto de datos $X \in \mathbb{R}^{n \times p}$ como una combinación de pocos elementos de un **diccionario aprendido** $D \in \mathbb{R}^{p \times k}$, donde $k$ es el número de átomos o componentes básicos.

### Representación básica

Cada muestra $x_i \in \mathbb{R}^p$ se aproxima como:

$$
x_i \approx D \alpha_i
$$

donde:

- $D = [d_1, d_2, ..., d_k]$ son los **átomos del diccionario** (columnas de $D$).
- $\alpha_i \in \mathbb{R}^k$ es un **vector de codificación dispersa** (muchos coeficientes son cero).

En conjunto, si $A = [\alpha_1, ..., \alpha_n]^\top \in \mathbb{R}^{n \times k}$ son todas las codificaciones:

$$
X \approx A D^\top
$$

### Problema de optimización

El entrenamiento de DictionaryLearning consiste en resolver:

$$
\min_{D, A} \; \frac{1}{2} \|X - A D^\top\|_F^2 + \lambda \|A\|_1
$$

sujeto a:

$$
\|d_j\|_2 \leq 1 \quad \forall j = 1, ..., k
$$

donde:

- $\|X - AD^\top\|_F^2$ mide el error de reconstrucción (norma de Frobenius).
- $\|A\|_1 = \sum_{i,j} |a_{ij}|$ promueve la **esparsidad** en las codificaciones.
- $\lambda > 0$ controla el nivel de regularización.

### Etapas del entrenamiento

1. **Codificación dispersa (Sparse Coding):**  
   Para un diccionario fijo $D$, se resuelve:

   $$
   \min_{\alpha_i} \; \frac{1}{2} \|x_i - D \alpha_i\|_2^2 + \lambda \|\alpha_i\|_1
   $$

   Esto se parece a un **problema Lasso** por cada muestra.

2. **Actualización del diccionario:**  
   Con las codificaciones $A$ fijas, se ajusta cada átomo $d_j$ para reducir el error de reconstrucción.  
   El proceso se repite iterativamente (similar al método de **alternating minimization**).

### Interpretación matemática

- Cada dato se expresa como **combinación lineal de pocos átomos**.
- El $\ell_1$-penalty sobre $A$ obliga a que **la mayoría de coeficientes sean cero**, garantizando **representaciones compactas**.
- El problema es **no convexo en conjunto ($D, A$)**, pero convexo en cada variable individualmente, lo que permite usar algoritmos alternados.

### Comparación con PCA

- PCA: busca ejes ortogonales que maximizan la varianza.
- DictionaryLearning: busca **átomos no necesariamente ortogonales**, adaptados a la estructura de los datos, y fuerza **esparsidad** en la representación.

---

## <b>Supuestos del Modelo</b>

Para que el modelo de **Dictionary Learning** funcione correctamente y produzca resultados confiables, se deben considerar los siguientes supuestos:

- **Naturaleza de los datos**  
  > El modelo está diseñado para trabajar con **datos numéricos continuos** en forma matricial.  
  > No es adecuado directamente para datos categóricos sin una transformación previa (como one-hot encoding).

- **Representación dispersa (sparse)**  
  > Se asume que las observaciones pueden representarse como una **combinación lineal dispersa** de un conjunto reducido de átomos (componentes del diccionario).  
  > Es decir, que solo unas pocas combinaciones son relevantes para explicar cada muestra.

- **Número de componentes adecuado**  
  > El número de átomos del diccionario (`n_components`) debe ser elegido considerando el problema.  
  > Si es demasiado pequeño, no capturará la estructura; si es demasiado grande, puede sobreajustar o perder interpretabilidad.

- **Independencia aproximada entre los átomos**  
  > Aunque no se exige ortogonalidad estricta (como en PCA), se espera que los componentes aprendidos no sean redundantes y capturen diferentes aspectos de los datos.

- **Escalamiento de variables**  
  > Es recomendable que los datos estén **escalados o normalizados** antes de aplicar el modelo.  
  > De lo contrario, variables en distintas escalas podrían dominar la descomposición.

- **Ruido en los datos**  
  > El modelo es relativamente robusto al ruido, pero **exceso de ruido puede impedir encontrar una representación dispersa útil**.

- **Adecuación del tipo de problema**  
  > Dictionary Learning se creó principalmente para problemas de:  
  > - **Procesamiento de imágenes** (compresión, denoising, reconocimiento de patrones).  
  > - **Señales** (audio, EEG, sensores).  
  > - **Datos de alta dimensión** donde se sospecha que existe una estructura latente dispersa.  
  > No es un método universal para cualquier dataset, sino específico para contextos donde la representación dispersa es razonable.

---

## <b>Interpretación</b>

Una vez entrenado el modelo de **Dictionary Learning**, la clave está en **interpretar la representación dispersa y el diccionario aprendido**. A continuación, se detallan los elementos más importantes:

### Diccionario ($D$)

El modelo aprende un conjunto de **átomos o componentes básicos** (columnas de $D$).  
Cada átomo representa un **patrón característico de los datos**.

- En imágenes: un átomo puede corresponder a bordes, texturas o estructuras repetitivas.  
- En señales: un átomo puede representar frecuencias o formas de onda características.  

> En esencia, el diccionario es una “base aprendida” a partir de los datos.

### Coeficientes dispersos ($A$)

Cada muestra $x_i$ se representa como una **combinación lineal dispersa** de los átomos:

$$
x_i \approx D \cdot a_i
$$

donde $a_i$ es el vector de coeficientes (la representación dispersa de $x_i$).

- La mayoría de los valores en $a_i$ son **ceros**.  
- Solo unos pocos coeficientes son diferentes de cero → indican **qué átomos son relevantes** para esa muestra.

> Interpretación: los coeficientes muestran **qué patrones del diccionario explican cada observación**.

### Error de reconstrucción

El modelo intenta minimizar la diferencia entre los datos originales $X$ y su reconstrucción $DA$:

$$
\| X - D A \|_2^2
$$

- Un error bajo significa que el diccionario y los coeficientes capturan bien la estructura de los datos.  
- Un error alto indica que el número de átomos es insuficiente o que los datos no se ajustan bien a una representación dispersa.

### Regularización y dispersidad

El modelo aplica una penalización de tipo L1 para inducir **sparsity** en los coeficientes:

$$
\min_{D, A} \| X - D A \|_2^2 + \lambda \| A \|_1
$$

- El parámetro $\lambda$ controla la dispersidad:  
  - $\lambda$ alto → más ceros en $A$ (más parsimonioso, menos detalle).  
  - $\lambda$ bajo → menos ceros (más detalle, pero menos interpretabilidad).

### Comparación con PCA

- En **PCA**, los componentes son ortogonales y cada observación usa todos los componentes.  
- En **Dictionary Learning**, los átomos **no son ortogonales** y cada observación usa solo unos pocos, lo que da una representación más **local e interpretable**.

### En resumen:

| Elemento           | Qué representa                                         |
|--------------------|---------------------------------------------------------|
| Diccionario $D$    | Patrones básicos aprendidos a partir de los datos       |
| Coeficientes $A$   | Qué patrones usa cada observación                       |
| Error de reconstrucción | Calidad de la representación dispersa              |
| $\lambda$ (sparsity) | Nivel de simplicidad vs. detalle en la representación |

---

## <b>Implementación en `scikit-learn`</b>

```python
from sklearn.decomposition import DictionaryLearning

model = DictionaryLearning(
    n_components=None,
    *,
    alpha=1,
    max_iter=1000,
    tol=1e-8,
    fit_algorithm='lars',
    transform_algorithm='omp',
    transform_n_nonzero_coefs=None,
    transform_alpha=None,
    n_jobs=None,
    code_init=None,
    dict_init=None,
    verbose=False,
    split_sign=False,
    random_state=None,
    positive_code=False,
    positive_dict=False,
    transform_max_iter=1000,
    callback=None,
    batch_size=None,
    shuffle=True,
    n_iter=None,
    transform_tol=1e-8
)

model.fit(X_train)
X_reduced = model.transform(X_test)
```

---

## <b>Parámetros Cruciales</b>

A continuación, se explican los hiperparámetros más importantes que afectan directamente la calidad de la descomposición y la representación aprendida por **DictionaryLearning**.

### n_components — Número de átomos en el diccionario

Define cuántos “patrones básicos” tendrá el diccionario.

- **Valor bajo** → puede no capturar la complejidad de los datos (**underfitting**).  
- **Valor alto** → puede sobreajustar e incluso aprender ruido (**overfitting**).

> Es el parámetro principal que determina la capacidad del modelo para representar los datos.

### alpha — Peso de la regularización (esparsidad)

Controla cuántos átomos del diccionario se utilizan para representar cada muestra.

- **Valor alto** → representaciones más **sparsas** (menos coeficientes distintos de cero).  
- **Valor bajo** → representaciones más densas, más cercanas a los datos originales.

> Balancea **compresión** vs **fidelidad en la reconstrucción**.

### max_iter — Iteraciones para el ajuste

Número máximo de pasos para optimizar el diccionario.

- **Bajo** → puede detenerse antes de converger.  
- **Alto** → mejor ajuste, pero mayor riesgo de sobreajuste si `alpha` es bajo.

### tol — Tolerancia de convergencia

Criterio para decidir cuándo detener la optimización.

- **Valor bajo** → el modelo busca soluciones más precisas.  
- **Valor alto** → el entrenamiento se detiene antes, con soluciones más aproximadas.

> Aunque no cambia radicalmente la solución, sí influye en el **grado de ajuste final**.

### fit_algorithm — Método para ajustar los códigos (sparse coding)

Indica cómo se calcula la representación esparsa de cada muestra:

- `'lars'` → más preciso, recomendado para alta esparsidad.  
- `'cd'` (coordinate descent) → más rápido y eficiente en datos grandes.  

> El método elegido puede producir ligeras diferencias en la reconstrucción.

### transform_algorithm — Método para transformar nuevos datos

Controla cómo se generan las representaciones esparsas en `transform()`:

- `'lasso_lars'`, `'lasso_cd'` → distintas variantes de Lasso.  
- `'omp'` (Orthogonal Matching Pursuit) → selecciona átomos paso a paso.  
- `'threshold'` → aplica un umbral sobre los coeficientes.

> La elección cambia el **tipo de esparsidad y precisión** de la representación.

### transform_alpha / transform_n_nonzero_coefs

- **transform_alpha** → regularización aplicada al transformar nuevos datos.  
- **transform_n_nonzero_coefs** → fija el número máximo de átomos activos por muestra.

> Determinan qué tan compactas o detalladas serán las representaciones para datos nuevos.

**Resumen gráfico mental:**

| Parámetro                 | Afecta...                               | Cuándo ajustarlo                              |
|----------------------------|-----------------------------------------|-----------------------------------------------|
| `n_components`            | Capacidad del diccionario               | Cuando hay underfitting u overfitting         |
| `alpha`                   | Nivel de esparsidad                     | Según si prefieres compresión o precisión     |
| `max_iter`                | Grado de ajuste                         | Si el modelo no converge                      |
| `tol`                     | Precisión de la optimización            | Para mayor o menor ajuste                     |
| `fit_algorithm`           | Cómo se calculan los códigos            | Según tamaño de datos y nivel de esparsidad   |
| `transform_algorithm`     | Cómo representar nuevos datos           | Según si prefieres rapidez o precisión        |
| `transform_alpha / n_nonzero_coefs` | Controlan esparsidad al transformar | Según nivel de compacidad deseado             |

---

## <b>Validaciones Numéricas Internas</b>

Cuando llamas al método `.fit()` de **DictionaryLearning** en `scikit-learn`, se inicia un proceso interno de **optimización alternada** para construir un diccionario de componentes y representaciones esparsas que mejor reconstruyan los datos.

### ¿Qué significa "entrenar" el modelo?

Entrenar el modelo significa **encontrar dos cosas al mismo tiempo**:

1. Un **diccionario $D$** (matriz de bases o átomos) de tamaño `(n_features × n_components)`.  
2. Unas **representaciones esparsas $C$** (también llamadas *códigos* o *coeficientes*) de tamaño `(n_samples × n_components)`.

El objetivo es que los datos originales $X$ puedan reconstruirse como:

$$
X \approx C \cdot D^T
$$

donde:
- $X$ = datos originales.  
- $C$ = representaciones esparsas.  
- $D$ = diccionario de átomos aprendidos.  

### ¿Qué función se minimiza?

La función de costo busca un balance entre **fidelidad en la reconstrucción** y **esparsidad en los coeficientes**:

$$
\mathcal{L}(C, D) = \frac{1}{2} \| X - C D^T \|_F^2 + \alpha \| C \|_1
$$

- Primer término: **error de reconstrucción** (qué tan bien $C \cdot D^T$ aproxima $X$).  
- Segundo término: **regularización L1** que fuerza esparsidad en $C$.  
- `alpha` → controla el grado de esparsidad.  

### ¿Qué hace internamente `.fit()`?

El entrenamiento se hace con un procedimiento iterativo de **optimización alternada**:

1. **Inicialización**
   - Se generan valores iniciales para $D$ (aleatorios u ortogonales).  

2. **Codificación esparsa (sparse coding)**
   - Para un diccionario fijo $D$, se calcula $C$ resolviendo un problema de regularización L1.  
   - Dependiendo de `fit_algorithm` se usa:
     - `'lars'` → Least Angle Regression (preciso).  
     - `'cd'` → Coordinate Descent (eficiente en grandes datos).  

3. **Actualización del diccionario**
   - Con $C$ fijo, se recalcula $D$ para minimizar el error de reconstrucción.  
   - Este paso se hace usando variantes de **Method of Optimal Directions (MOD)** o actualizaciones de tipo **least squares**.  

4. **Iteración alternada**
   - Se repiten pasos 2 y 3 hasta que:
     - Se alcanza el número máximo de iteraciones (`max_iter`), o  
     - La mejora en el error es menor que la tolerancia (`tol`).  

### ¿Qué devuelve al final?

Después de entrenar:

- `components_` → el diccionario aprendido ($D$).  
- `n_iter_` → número de iteraciones realizadas.  
- `error_` (si está disponible) → error de reconstrucción.  
- `transform(X)` → genera las representaciones esparsas $C$ de nuevos datos.  

### Importante

- El modelo **no siempre converge** si `alpha` es demasiado bajo (representaciones muy densas).  
- Para datasets grandes, es recomendable:
  - Usar `fit_algorithm='cd'`.  
  - Ajustar `max_iter` y `tol`.  

- La elección de `transform_algorithm` afecta cómo se representarán **nuevos datos** (Lasso, OMP, Threshold, etc.).

### En resumen

Entrenar un **DictionaryLearning** significa resolver un **problema de factorización matricial con restricción de esparsidad**:  
se aprende un conjunto de átomos $D$ y códigos esparsos $C$ que permiten reconstruir $X$ de manera compacta y eficiente.

---

## <b>Casos de uso</b>

El algoritmo de **Dictionary Learning** es especialmente útil cuando se busca una **representación más compacta y esparsa de los datos**. Aunque existen técnicas más modernas como autoencoders o NMF, este modelo sigue siendo muy valioso en distintos escenarios.

### Compresión de imágenes y señales

- Permite representar imágenes como combinaciones esparsas de "átomos" (patrones locales).  
- Reduce almacenamiento manteniendo las características importantes.  
- Útil en sistemas de transmisión con restricciones de ancho de banda.  

> Ejemplo: compresión de rostros en bases de datos biométricos.

### Eliminación de ruido (denoising)

- Al usar solo los átomos más relevantes, se eliminan detalles aleatorios o ruido gaussiano.  
- Muy utilizado en **procesamiento de imágenes médicas** (MRI, radiografías) o señales de audio.  

> El ruido rara vez se representa con pocos átomos, mientras que la estructura sí.

### Reconocimiento de patrones (faces, objetos, voz)

- Cada clase (persona, objeto, palabra) puede aprender su propio subconjunto de átomos.  
- El modelo puede clasificar proyectando nuevas muestras en esos diccionarios.  

> Ejemplo: reconocimiento facial basado en descomposición esparsa.

### Análisis de datos de alta dimensión

- En datos con muchas variables, Dictionary Learning crea una **base reducida y significativa**.  
- Mejor que PCA cuando se necesita **esparsidad** y no solo reducción de dimensión.  

> Ideal en bioinformática, señales cerebrales (EEG), o espectros químicos.

### Modelado de texto y NLP

- Similar a topic modeling: los documentos se representan como combinaciones esparsas de “tópicos” (átomos).  
- Puede ser una alternativa a LDA en ciertos casos.  

> Útil en clasificación de documentos y búsqueda semántica.

### ¿Cuándo *NO* usar Dictionary Learning?

- Cuando los datos son pequeños y simples → un PCA o regresión básica suele ser suficiente.  
- Cuando se requiere **predicción directa** y no solo representación → modelos supervisados funcionan mejor.  
- En datos secuenciales muy largos → redes neuronales recurrentes o transformadores son superiores.  

### Conclusión

> **Dictionary Learning** es la opción correcta cuando necesitas **representaciones esparsas, interpretables y compactas** que capturen la estructura latente de los datos.  
> Se destaca en compresión, reducción de ruido y reconocimiento de patrones, donde los datos se explican mejor como **combinaciones de pocos componentes base**.

---

## <b>Profundización matemática</b>

Esta sección profundiza en la **matemática y los algoritmos** que usa `DictionaryLearning` para aprender un diccionario $D$ y códigos esparsos $C$. Aquí se detallan las formulaciones, los métodos de *sparse coding*, las actualizaciones del diccionario (MOD, K-SVD), operadores proximales (ISTA/FISTA), condiciones de recuperación y notas sobre convergencia.

### Problema central (recordatorio)

Dado $X \in \mathbb{R}^{n \times p}$ (n muestras, p características) se busca:

$$
\min_{D \in \mathbb{R}^{p\times k},\, C \in \mathbb{R}^{n\times k}} \; \frac{1}{2}\|X - C D^\top\|_F^2 + \lambda \|C\|_1
\quad\text{s.t.}\quad \|d_j\|_2 \le 1\ \forall j
$$

- $D=[d_1,\dots,d_k]\in\mathbb{R}^{p\times k}$ (átomos en columnas).  
- $C=[\alpha_1^\top;\dots;\alpha_n^\top]\in\mathbb{R}^{n\times k}$ (cada fila $\alpha_i$ codifica $x_i$).  
- $\lambda$ controla esparsidad.

El problema **no es convexo** en $(D,C)$ conjuntamente, pero **es convexo** en cada bloque por separado (D fijo $\Rightarrow$ convexa en $C$, y viceversa).

### 1) Sparse coding (codificación esparsa) — por muestra

Para cada muestra $x\in\mathbb{R}^p$ (fila de $X$), la sub-tarea es:

$$
\alpha^\star = \arg\min_{\alpha\in\mathbb{R}^k} \frac{1}{2}\|x - D\alpha\|_2^2 + \lambda \|\alpha\|_1
$$

Métodos comunes:

#### a) Coordinate Descent (CD)** — actualización por coordenadas (Lasso)
Para cada componente $j$:

1. Calcula el residuo parcial sin la j-ésima contribución:
   $$
   r^{(j)} = x - D\alpha + d_j \alpha_j
   $$
2. Producto interno:
   $$
   z_j = d_j^\top r^{(j)}
   $$
3. Actualiza con **soft-thresholding**:
   $$
   \alpha_j \leftarrow S\!\left(\frac{z_j}{\|d_j\|_2^2},\; \frac{\lambda}{\|d_j\|_2^2}\right)
   $$
   donde $S(z,t)=\operatorname{sign}(z)\max(|z|-t,0)$.

CD converge rápido cuando $D$ tiene columnas normalizadas y es eficiente para problemas de Lasso.

#### b) ISTA / FISTA (proximal gradient)

- Paso de gradiente + prox:

  $$
  \alpha^{(t+1)} = S\!\left(\alpha^{(t)} + \frac{1}{L}D^\top (x - D\alpha^{(t)}),\; \frac{\lambda}{L}\right)
  $$

  donde $L=\|D^\top D\|_2$ (constante Lipschitz del gradiente).

- **FISTA** añade aceleración (momentum) y tiene convergencia $O(1/t^2)$ frente a $O(1/t)$ de ISTA.

#### c) Orthogonal Matching Pursuit (OMP) — método greedy
- Inicializa $r = x$, soporte $S=\varnothing$.
- Repetir hasta criterio (p. ej. $m$ átomos o residuo pequeño):
  1. Seleccionar $j^\star = \arg\max_j |d_j^\top r|$ (mayor correlación).
  2. Añadir $j^\star$ a $S$ y resolver $\min_{\alpha_S}\|x - D_S \alpha_S\|_2^2$ (LS sobre columnas seleccionadas).
  3. Actualizar residuo $r = x - D_S \alpha_S$.
- OMP da soluciones con exactamente $m$ átomos (control explícito de esparsidad).

#### d) LARS (Least Angle Regression) / Homotopy
- LARS traza el camino de soluciones para Lasso a medida que varía $\lambda$; útil para `lars` cuando se desea la trayectoria completa o soluciones exactas para pocos parámetros activos.

### 2) Actualización del diccionario

Con $C$ fijo, $D$ se obtiene resolviendo problema de mínimos cuadrados con restricción de norma:

$$
\min_{D: \|d_j\|\le1} \; \|X - C D^\top\|_F^2
\quad\Longleftrightarrow\quad
\min_{D} \|X^\top - D C^\top\|_F^2
$$

Solución (sin restricción de norma) en forma cerrada (MOD):

$$
D = X^\top C (C^\top C)^{-1}
$$

Luego se normalizan las columnas: $d_j \leftarrow d_j / \max(1,\|d_j\|_2)$ para satisfacer $\|d_j\|_2\le1$.

#### a) MOD (Method of Optimal Directions)  
- Actualización global con la fórmula anterior cuando $C^\top C$ invertible. Simple y rápido para batch.

#### b) K-SVD (actualización por átomo con SVD)
Para cada átomo $d_j$:
1. Define el residuo excluyendo $d_j$:
   $$
   R_j = X - \sum_{l\ne j} \alpha_{\cdot,l} d_l^\top
   $$
   (donde $\alpha_{\cdot,l}$ es la columna l de $C^\top$, equivalentes).
2. Restrinje $R_j$ a las muestras que usan el átomo $j$ (índices $I_j = \{i: \alpha_{i,j}\ne 0\}$): $R_j^{(I_j)}$.
3. Calcula la SVD de $R_j^{(I_j)}$ y actualiza:
   $$
   U \Sigma V^\top = \operatorname{SVD}(R_j^{(I_j)})
   $$
   - Nuevo átomo $d_j = U[:,1]$ (primer vector singular).
   - Nuevos coeficientes $\alpha_{I_j,j} = \Sigma_{1,1} V[:,1]^\top$.
K-SVD mejora localmente la reconstrucción y suele dar mejores átomos pero es más costoso.

### 3) Normalización y ambigüedad de escala

Existe ambigüedad escalar entre $D$ y $C$: $(c \cdot d, \alpha / c)$ produce la misma reconstrucción. Por eso se impone $\|d_j\|_2 \le 1$ o se normaliza columnas tras cada actualización.

### 4) Operador proximal y soft-thresholding

La prox del término $\lambda \|\alpha\|_1$ es el **soft-threshold**:

$$
\operatorname{prox}_{\lambda\|\cdot\|_1}(z) = S(z,\lambda) = \operatorname{sign}(z)\max(|z|- \lambda, 0)
$$

Este operador aparece explícitamente en ISTA/FISTA y en las actualizaciones por coordenadas (soft-thresholding closed form).

### 5) Condiciones de recuperación y unicidad

- **Coherencia mutua** del diccionario:
  $$
  \mu(D) = \max_{i\ne j} |d_i^\top d_j|.
  $$
  Si la representación tiene sparsity $s$ y cumple
  $$
  s < \frac{1}{2}\left(1 + \frac{1}{\mu(D)}\right),
  $$
  entonces la solución escasa es única y algoritmos greedy (OMP) o L1 (Lasso) pueden recuperarla exactamente bajo ciertas condiciones.

- **RIP (Restricted Isometry Property)** y otras condiciones sufientes (más técnicas) garantizan recuperación estable frente a ruido.

### 6) Convergencia y propiedades numéricas

- El esquema **alternating minimization** (fijar D ↔ fijar C) garantiza que la función objetivo no aumenta en cada paso, por lo que la secuencia de valores objetivo es monótona decreciente y converge a un punto crítico (posible mínimo local). **No garantiza mínimo global** (no convexo globalmente).
- Con condiciones adicionales (incoherencia, sparsity), y con buenos inicializadores, se pueden obtener soluciones cercanas al óptimo global.
- En práctica: usar normalización, buen $\lambda$, y múltiples inicializaciones para mitigar mínimos locales.

### 7) Variantes online / batch

- **Batch (MOD, K-SVD)**: usan todo $X$ cada iteración → más preciso, menos escalable.
- **Online Dictionary Learning (Mairal et al.)**: minimiza expectativa empírica usando muestras o mini-lotes y actualizaciones estocásticas para $D$; converge a un punto estacionario y escala a grandes datasets.
  - Objetivo esperado:
    $$
    \min_D \mathbb{E}_x\left[\min_\alpha \frac{1}{2}\|x - D\alpha\|_2^2 + \lambda\|\alpha\|_1\right].
    $$

### 8) Elección de parámetros — implicaciones matemáticas

- $\mathbf{n\_components}$: dimensiona el espacio de representación. Matemáticamente controla el subespacio (o combinación de subespacios) que $D$ puede generar.
- $\mathbf{\lambda}$ (alpha): directamente controla la penalidad L1; aumenta la cardinalidad cero de $\alpha$ y por tanto mejora condicionalmente la identificabilidad del soporte.
- $\mathbf{\text{transform\_n\_nonzero\_coefs}}$ vs $\mathbf{\text{transform\_alpha}}$: uno fija la cardinalidad (constraint $|\operatorname{supp}(\alpha)|\le s$), el otro impone penalización; ambos cambian la estructura matemática del subproblema (combinatorial vs convex).

### 9) Complejidad computacional (apócope)

- Sparse coding por muestra (CD/ISTA): costo aproximado $O(k p \cdot t)$ donde $t$ son iteraciones internas.  
- OMP greedy (m no cero): $O(m p k)$ por muestra si se implementa eficientemente.  
- Actualización de diccionario (MOD): requiere invertir $C^\top C$ ($O(k^3)$) y multiplicaciones $O(pk^2 + pkn)$ en implementaciones ingenuas.

### 10) Resumen (intuición matemática)

- `DictionaryLearning` = resolver una factorización $X \approx C D^\top$ con **penalización L1** sobre $C$ para promover esparsidad.  
- Algoritmos alternan entre **resolver Lasso/OMP** por cada muestra y **minimizar por mínimos cuadrados** para $D$ (MOD) o actualizar átomo-a-átomo (K-SVD).  
- Matemáticamente mezcla herramientas de optimización convexa (subproblemas de Lasso) y métodos algebraicos (SVD, LS) dentro de un esquema no convexo global (alternating minimization).

---

## <b>Recursos para profundizar</b>

**Libros**  
- *Sparse and Redundant Representations* – Michael Elad  
- *Foundations of Data Science* – Blum, Hopcroft, Kannan  
- *Pattern Recognition and Machine Learning* – Christopher Bishop  

**Cursos**  
- Coursera – *Machine Learning Specialization* (Andrew Ng)  
- MIT OpenCourseWare – *Machine Learning*  
- FastAI – *Practical Deep Learning for Coders*  
- YouTube – *StatQuest with Josh Starmer* (explica conceptos como PCA y descomposición de diccionarios de forma clara)  

**Documentación oficial**  
- [scikit-learn: DictionaryLearning](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html)

---