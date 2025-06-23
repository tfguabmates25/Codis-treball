# Representacions vectorials de text en problemes de classificació

## Introducció

Aquest repositori recull el codi desenvolupat en el marc del Treball de Fi de Grau de Matemàtiques centrat en l'estudi de diverses tècniques de vectorització de textos per a tasques de classificació. L'objectiu principal del treball és formalitzar i analitzar matemàticament el procés de **representació vectorial dels documents textuals**, comparant tres mètodes fonamentals: **TF-IDF**, **LSA** i **LDA**. A més de la base teòrica, el projecte aplica aquestes representacions a diversos conjunts de dades de text (correus spam, missatges de Twitter, col·lecció *20 Newsgroups*) per avaluar-ne l'eficàcia en classificació binària mitjançant models supervisats.

Amb aquest enfocament combinat, el treball proporciona una comparativa entre un model basat en el vocabulari original (TF-IDF) i dos models latents o "temàtics" que redueixen la dimensionalitat (LSA i LDA). El codi disponible permet reproduir els experiments realitzats, incloent el preprocessament dels textos, la transformació vectorial i l'entrenament/avaluació de classificadors sobre cada representació.

## Representacions vectorials utilitzades

En aquesta secció es descriuen de manera clara i formal les tècniques de representació vectorial implementades en el treball. Cada mètode converteix documents de text en vectors numèrics seguint enfocaments diferents, ja sigui basant-se en les paraules originals o en temes latents inferits dels textos.

### TF-IDF (Term Frequency - Inverse Document Frequency)

**TF-IDF** és una tècnica clàssica de representació de textos que assigna a cada terme un pes numèric proporcional a la seva importància en un document concret i en el corpus global. Aquesta importància es calcula com el producte de dues components:

- **Freqüència del terme (TF)**: mesura la recurrència del terme $t$ dins del document $d$:

$$
tf(t,d) = \frac{f(t,d)}{\max {f(t',d) \mid t' \in d}}
$$

on \( f_{t,d} \) és la freqüència del terme $t$ al document $d$ i el denominador és la el màxim de les freqüències de tots els termes del document.

- **Freqüència inversa de document (IDF)**: mesura la raresa del terme en el conjunt de documents $D$

$$
\mathrm{idf}(t, D) = \log\left( \frac{|D|}{ |\{ d \in D : t \in d \}|} \right)
$$

on $|D|$ és el nombre total de documents i el denominador és el nombre de documents en què apareix el terme $t$. El valor 1 s’afegeix per evitar divisions per zero.

- **Pes TF-IDF**: és el producte de les dues components anteriors:

$$
\mathrm{tf\text{-}idf}(t, d) = \mathrm{tf}(t, d) \cdot \mathrm{idf}(t, D)
$$

D’aquesta manera, els termes molt específics (freqüents en un document però poc freqüents al conjunt de documents) obtenen un pes alt. En canvi, termes molt comuns en tots els documents són penalitzats.

Cada document queda representat com un vector en un espai de dimensionalitat igual a la mida del vocabulari, on cada component del vector és el valor TF-IDF associat a un terme. Aquesta representació conserva la granularitat original del text i és especialment adequada per a algoritmes de classificació que poden gestionar espais d’alta dimensionalitat.


### LSA (Latent Semantic Analysis via Truncated SVD)

**LSA** o *Anàlisi Semàntica Latent* és un mètode algebraic que permet obtenir representacions latents i més compactes dels documents textuals. El procés comença construint una matriu terme-document $A \in \mathbb{R}^{m \times n}$, on cada fila representa un terme, cada columna un document, i les entrades són pesos (habitualment valors TF-IDF) que reflecteixen la importància del terme al document.

A continuació, s’aplica una **Descomposició en Valors Singulars (SVD)** sobre la matriu $A$:

$$
A = U \Sigma V^T
$$

on:

- $U \in \mathbb{R}^{m \times r}$ conté els vectors singulars esquerres (relacionats amb els termes),
- $\Sigma \in \mathbb{R}^{r \times r}$ és una matriu diagonal amb els valors singulars no negatius ordenats descendentment,
- $V^T \in \mathbb{R}^{r \times n}$ conté els vectors singulars drets (relacionats amb els documents),
- i $r$ és el rang de la matriu original $A$.

Per reduir la dimensionalitat, es trunca aquesta descomposició retinguent només els $k$ primers components (amb $k\ll r$):

$$
A_k = U_k \Sigma_k V_k^T
$$

Segons el **teorema d’Eckart–Young**, $A_k$ és la millor aproximació de rang $k$ a la matriu $A$ en norma Frobenius, i conserva les estructures semàntiques principals del corpus.

En aquest espai latent de $k$ dimensions:

- Cada document s'expressa com $d_i=\( \Sigma_k v_i \)$, $v_i$ dennota la i-éssima fila de $V^T_k$

Aquesta nova representació projecta els documents i termes en un espai semàntic reduït, on documents amb significat similar (encara que no comparteixin paraules literals) es troben més propers entre si. Així, **LSA capta relacions latents i sinònimes entre paraules i documents**, oferint representacions vectorials més compactes que poden millorar l’eficiència computacional i la generalització del model.

### LDA (Latent Dirichlet Allocation)

**LDA** o *Assignació Latent de Dirichlet* és un model probabilístic generatiu que representa documents mitjançant temes latents. L’enfocament fonamental del model és considerar que cada document $d$ és generat per una combinació de $k$ temes latents, i que cada tema $z_k$ és una distribució de probabilitat sobre les paraules del vocabulari.

El model assumeix el següent procés generatiu:

1. Per a cada tema $k = 1, \dots, K$, es genera una distribució de paraules $\phi^k \sim \text{Dir}(\beta)$, on $\phi^k \in \mathbb{R}^V$ i $V$ és la mida del vocabulari.
2. Per a cada document $d$:
   - Es genera una distribució de temes $\theta_d \sim \text{Dir}(\alpha)$, on $\theta_d \in \mathbb{R}^K$.
   - Per a cada paraula $w_{n}$ del document $d$:
     - Es tria un tema latent $z_{n}\sim \text{Multinomial}(\theta_d)$,
     - Es tria una paraula $w_{n} \sim \text{Multinomial}(\phi^{z_{n}})$.

En resum, les **variables latents** són:

- $\theta_d$: vector de probabilitats de temes per al document $d$,
- $z_{n}$: assignació de tema a la \( n \)-èsima paraula del document $d$,
- $\phi^k$: distribució de paraules del tema $k$.

Durant el procés d’inferència (per exemple, mitjançant *mostreig de Gibbs* o *mètodes variacionals*), s’estimen les distribucions a posteriori $p(\theta_d \mid w_d)$ i $p(\phi^k \mid w_d)$.

Per tant, la representació final d’un document és el vector $\theta_d \in \mathbb{R}^K$, que expressa la proporció estimada de cada tema latent en el document. Aquesta **representació temàtica** redueix la dimensionalitat del document respecte de l'espai original de paraules i capta **estructures semàntiques globals** del corpus.

---

**Nota:** Tant **LSA** com **LDA** són mètodes de reducció de dimensionalitat que extreuen factors semàntics subjacents als documents. LSA utilitza tècniques algebraiques com la SVD truncada, mentre que LDA utilitza un model probabilístic generatiu amb distribucions de Dirichlet. En canvi, **TF-IDF** representa els documents directament a partir del vocabulari observable, sense considerar temes latents. Segons els resultats obtinguts en aquest treball, encara que **TF-IDF** sol proporcionar el millor rendiment predictiu, **LSA** i **LDA** ofereixen representacions més compactes i semànticament interpretables, especialment útils en conjunts de dades amb redundància o correlació lèxica alta.

## INSTRUCCIONS PER A CODI
### Requisits del sistema

* **Python 3.8+**: Es recomana tenir instal·lada una versió recent de Python (3.8 o superior).
* **Llibreries Python**: Cal instal·lar les següents dependències (amb versions equivalents a les utilitzades en el TFG):

  * pandas (manipulació de dades)
  * numpy (operacions numèriques)
  * scikit-learn (vectorització TF-IDF, SVD, LDA, mètriques i validació)
  * NLTK (processament de text: tokenització i lematització; assegurar la disponibilitat de recursos com *punkt* i *wordnet*)
  * xgboost (implementació del classificador Gradient Boosting)
* **Altres**: No es requereix maquinari especial; l'execució és factible en CPU. Es necessita connexió a Internet la primera vegada que s'executin alguns scripts per tal de descarregar dades:

  * El conjunt de dades de SMS *spam* es carrega directament des d'una URL pública.
  * Alguns recursos de *NLTK* (com el model de tokenització) es descarregaran automàticament en cridar nltk.download(...) dins dels scripts
### Execució dels scripts pas a pas

1. **Preparar l'entorn:** Descarregueu o cloneu aquest repositori al vostre ordinador. Assegureu-vos d'haver instal·lat tots els paquets requerits esmentats en la secció de requisits. Si cal, utilitzeu pip o conda per instal·lar les dependències.
2. **Executar els experiments amb *Spam Dataset*:** Des de la línia de comandes, navegueu fins al directori spam/ i executeu els tres scripts d'aquesta carpeta:

   * python spam_TFIDF.py
   * python spam_LSA.py
   * python spam_LDA.py

   Aquests scripts carregaran automàticament el conjunt de dades de SMS spam (des d'un URL públic), preprocessaran els missatges (eliminant emojis, aplicant lematització, etc.) i entrenaran un model de classificació per avaluar la representació. Cada script realitza validació creuada (5-folds) entrenant un model de Gradient Boosting (XGBoost) i mostrant en pantalla la mètrica AUC-ROC obtinguda per diiferents folds.
3. **Executar els experiments amb *Twitter Dataset*:** Abans d'executar aquests scripts, assegureu-vos que el fitxer de dades data/twitter_redut_Dataset.csv es troba accessible. Després, des de la línia de comandes, executeu els tres scripts al directori twitter/:

   * python twitter_TFIDF.py
   * python twitter_LSA.py
   * python twitter_LDA.py

   Cadascun d'aquests scripts llegirà el CSV de tuits, aplicarà la vectorització corresponent (TF-IDF, LSA amb SVD truncada, o LDA amb un nombre de temes k especificat al codi) i entrenarà un classificador supervisat (en aquests experiments s'utilitza principalment **Regressió Logística**). Igualment, s'efectua validació creuada estratificada (5-folds) i es calcula l'**AUC-ROC** mitjana per comparar el rendiment entre mètodes.
4. **Executar els experiments amb *20 Newsgroups Dataset*:** Des del directori newsgroups/, executeu els scripts:

   * python newsgroups_TFIDF.py
   * python newsgroups_LSA.py
   * python newsgroups_LDA.py

   Aquests scripts descarregaran el corpus de *20 Newsgroups* (si no es troba ja en memòria cau) mitjançant *scikit-learn*. Es convertirà el problema en una classificació binària definint una etiqueta objectiu (per exemple, identificant documents de temàtica esportiva). Cada script generarà la representació vectorial pertinent per als documents (TF-IDF, reducció SVD a components latents, o distribucions de temes via LDA) i entrenarà un model de classificació (es fa servir **XGBoost** en aquest conjunt). Finalment es mostrarà l'AUC-ROC mitjana obtinguda en la validació creuada.
5. **Anàlisi de resultats:** Un cop executats tots els scripts, es poden comparar les mètriques AUC-ROC per veure quina representació vectorial ha funcionat millor en cada conjunt de dades. En general, s'observa que la representació basada en TF-IDF tendeix a oferir el millor rendiment predictiu, però les representacions latents (LSA, LDA) aconsegueixen resultats propers amb l'avantatge de reduir la dimensionalitat i captar relacions semàntiques latents.


