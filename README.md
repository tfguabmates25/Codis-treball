# Representacions vectorials de text en problemes de classificació

## Introducció

Aquest repositori recull el codi desenvolupat en el marc del Treball de Fi de Grau de Matemàtiques centrat en l'estudi de diverses tècniques de vectorització de textos per a tasques de classificació. L'objectiu principal del treball és formalitzar i analitzar matemàticament el procés de **representació vectorial dels documents textuals**, comparant tres mètodes fonamentals: **TF-IDF**, **LSA** i **LDA**. A més de la base teòrica, el projecte aplica aquestes representacions a diversos conjunts de dades de text (correus spam, missatges de Twitter, col·lecció *20 Newsgroups*) per avaluar-ne l'eficàcia en classificació binària mitjançant models supervisats.

Amb aquest enfocament combinat, el treball proporciona una comparativa entre un model basat en el vocabulari original (TF-IDF) i dos models latents o "temàtics" que redueixen la dimensionalitat (LSA i LDA). El codi disponible permet reproduir els experiments realitzats, incloent el preprocessament dels textos, la transformació vectorial i l'entrenament/avaluació de classificadors sobre cada representació.

## Representacions vectorials utilitzades

En aquesta secció es descriuen de manera clara i formal les tècniques de representació vectorial implementades en el treball. Cada mètode converteix documents de text en vectors numèrics seguint enfocaments diferents, ja sigui basant-se en les paraules originals o en temes latents inferits dels textos.

### TF-IDF (Term Frequency - Inverse Document Frequency)

**TF-IDF** és una tècnica clàssica de representació de textos que assigna a cada terme un pes numèric proporcional a la seva importància en un document concret i en el corpus global. Aquesta importància es calcula com el producte de dues components:

- **Freqüència del terme (TF)**: mesura la recurrència del terme \( t \) dins del document \( d \):

$$
\mathrm{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}
$$

on \( f_{t,d} \) és la freqüència del terme \( t \) al document \( d \), i el denominador és la suma de freqüències de tots els termes del document.

- **Freqüència inversa de document (IDF)**: mesura la raresa del terme en el conjunt de documents \( D \):

$$
\mathrm{IDF}(t, D) = \log\left( \frac{N}{1 + |\{ d \in D : t \in d \}|} \right)
$$

on \( N \) és el nombre total de documents i el denominador és el nombre de documents en què apareix el terme \( t \). El valor 1 s’afegeix per evitar divisions per zero.

- **Pes TF-IDF**: és el producte de les dues components anteriors:

$$
\mathrm{TF\text{-}IDF}(t, d, D) = \mathrm{TF}(t, d) \cdot \mathrm{IDF}(t, D)
$$

D’aquesta manera, els termes molt específics (freqüents en un document però poc freqüents al conjunt de documents) obtenen un pes alt. En canvi, termes molt comuns en tots els documents són penalitzats.

Cada document queda representat com un vector en un espai de dimensionalitat igual a la mida del vocabulari, on cada component del vector és el valor TF-IDF associat a un terme. Aquesta representació conserva la granularitat original del text i és especialment adequada per a algoritmes de classificació que poden gestionar espais d’alta dimensionalitat.


### LSA (Latent Semantic Analysis via Truncated SVD)

**LSA** o *Anàlisi Semàntica Latent* és un mètode algebraic que permet obtenir representacions latents i més compactes dels documents textuals. El procés comença construint una matriu terme-document \( A \in \mathbb{R}^{m \times n} \), on cada fila representa un terme, cada columna un document, i les entrades són pesos (habitualment valors TF-IDF) que reflecteixen la importància del terme al document.

A continuació, s’aplica una **Descomposició en Valors Singulars (SVD)** sobre la matriu \( A \):
$$
A = U \Sigma V^T
$$

on:

- \( U \in \mathbb{R}^{m \times r} \) conté els vectors singulars esquerres (relacionats amb els termes),
- \( \Sigma \in \mathbb{R}^{r \times r} \) és una matriu diagonal amb els valors singulars no negatius ordenats descendentment,
- \( V^T \in \mathbb{R}^{r \times n} \) conté els vectors singulars drets (relacionats amb els documents),
- i \( r \) és el rang de la matriu original \( A \).

Per reduir la dimensionalitat, es trunca aquesta descomposició retinguent només els \( k \) primers components (amb \( k \ll r \)):

$$
A_k = U_k \Sigma_k V_k^T
$$

Segons el **teorema d’Eckart–Young**, \( A_k \) és la millor aproximació de rang \( k \) a la matriu \( A \) en norma Frobenius, i conserva les estructures semàntiques principals del corpus.

En aquest espai latent de \( k \) dimensions:

- Cada document $d_i=\( \Sigma_k v_i \)$, $v_i$ dennota la i-éssima fila de $V^T_k$

Aquesta nova representació projecta els documents i termes en un espai semàntic reduït, on documents amb significat similar (encara que no comparteixin paraules literals) es troben més propers entre si. Així, **LSA capta relacions latents i sinònimes entre paraules i documents**, oferint representacions vectorials més compactes que poden millorar l’eficiència computacional i la generalització del model.
