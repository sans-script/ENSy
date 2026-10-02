# ENSy v2.1.3 — Revisão científica e validação da implementação

**Escopo:** revisão do plano `ENSy_v2.1.2_plano_implementacao.md` contra as fontes primárias e contra o código, seguida da implementação de referência (`ensy/`, `app.py`, frontend) e da validação em teste. Data: 2026-10-01.

**Fontes consultadas:** LEE; VERLEYSEN (2008, AISTATS/FSDM, JMLR W&CP 4) — texto integral; documento de escopo do TCC (Google Drive). O artigo de 2009 (*Neurocomputing*) não pôde ser baixado (servidor da UCLouvain inacessível) e o caderno do NotebookLM exige login Google, sem conector disponível — as afirmações sobre a versão de 2009 permanecem como citadas no plano, não verificadas por esta revisão.

---

## 1. Resultado da revisão — o que estava certo

| Item | Verificação |
|---|---|
| Definição de $Q$, blocos $UL/UR/LL/LR$, $D/LT/UT$ | Confere com Lee & Verleysen (2008), Sec. 2 |
| $M_T(K)$, $M_C(K)$ e $G_K$ | Idênticos a `sklearn.manifold.trustworthiness` (erro $<10^{-12}$, $K\in\{5,10,15,20\}$) |
| Nº de K-intrusões severas = nº de K-extrusões severas | Confere com Eq. (8) de 2008; testado |
| $Q_{NX} = U_P+U_N+U_X = \text{LCMC} + K/(N-1)$ | Confere com Eq. (13) de 2008; testado contra a definição direta |
| **Valor numérico** de $B_{NX}(K)$ da v2.1.2, positivo = intrusivo | **Confirmado empiricamente** (ver 2.1) |
| Decomposição local, $\text{mean}_i B_{NX,i} = B_{NX}$ | Exata por construção; testada (erro $<10^{-12}$) |
| Identidade $|F_K(i)| = |M_K(i)|$ (Sec. 9.4) | Testada |
| $\|\hat u-\hat v\|^2 = 2(1-\cos\theta)$ | Correto; consequência: após L2, postos por cosseno = postos euclidianos (testado) |

## 2. Erros e lacunas encontrados no plano

### 2.1 $B_{NX}$: a convenção certa, pela razão errada

O plano atribui a correção da v2.1.2 às Eq. 22–23 do artigo de 2009. No precursor de 2008 (mesmos autores, mesmo arcabouço):

- o texto põe as mild K-intrusions em $LT_K$ e as mild K-extrusions em $UT_K$;
- mas a Eq. (11) define $U_N$ somando sobre $UT_K$ e $U_X$ sobre $LT_K$, e a Eq. (14) define $B_{NX} = U_N - U_X$;
- logo, numericamente, $B_{NX} = \frac{1}{KN}(\sum_{UT_K} - \sum_{LT_K})$ — **o mesmo número** que a v2.1.2 obtém com $U_N$ sobre $LT_K$ e $B_{NX} = U_X - U_N$;
- o mesmo artigo afirma que $B_{NX} = W^{0,0}_N - W^{0,0}_X$, que tem o **sinal oposto** ao das suas Eqs. 11+14.

A fonte é internamente inconsistente quanto a rótulos. Por isso o sinal foi ancorado em algo inequívoco: o próprio artigo define $B_{TC} = M_C - M_T$ como indicador de comportamento, positivo para mapas intrusivos (o NLM "esmaga" a variedade e é intrusivo). Teste: esfera oca projetada ortograficamente (esmagamento, intrusivo) → $B_{TC}>0$ **e** $B_{NX}>0$; esfera desenrolada (rasgo, extrusivo) → ambos $<0$. Recomendação para o texto do TCC: definir $B_{NX}$ **pelos blocos**, não apenas por "$U_X - U_N$".

### 2.2 Casos sintéticos B e C com sinais trocados

A v2.1.2 dizia: "Caso B — promoção intrusiva → $B_{NX}<0$"; "Caso C — degradação extrusiva → $B_{NX}>0$". Isso contradiz a regra "positivo = intrusivo" do próprio plano. Além disso, por permutação não é possível criar uma mild intrusion isolada: promover um item de $m$ para 1 rebaixa os $m-1$ itens acima dele. Casos corrigidos (todos aprovados):

| Caso | Construção | Resultado exato |
|---|---|---|
| A | $\hat r = r$ | $B_{NX}=0$, $M_T=M_C=Q_{NX}=1$ |
| B | item do posto $m\le K$ promovido a 1 | $B_{NX,i} = (m-2)/K > 0$ (intrusivo); demais pontos 0 |
| C | item do posto 1 rebaixado a $m\le K$ | $B_{NX,i} = -(m-2)/K < 0$ (extrusivo) |
| troca | dois postos trocados dentro de $K$ | $U_N = U_X = 1/(KN)$, $B_{NX}=0$ |
| intrusão severa | item distante (posto 40) promovido a 1 | $U_{N,i}=0$, $U_{X,i}=(K-1)/K$ |

### 2.3 Outras correções de texto (já aplicadas no plano)

- Limite local exato: $|B_{NX,i}(K)| \le (K-1)/K$, e não "aproximadamente $\pm1$".
- "Um ponto pode sofrer muitas intrusões severas e ter $B_{NX,i}\approx0$": só é verdade se os intrusos caem no fim da vizinhança. Um intruso no posto 1 produz $U_{X,i}=(K-1)/K$.
- Removidos os exemplos "t-SNE em áreas esparsas" e "Isomap sob colapso": não verificados, e antecipavam o resultado.
- Referências acrescentadas: LEE; VERLEYSEN (2008), CHEN; BUJA (2009, LCMC), VENNA; KASKI (2001, origem de T&C). A citação de VENNA; KASKI (2006) difere entre o plano (*Neural Networks*) e o escopo (ESANN) — unificar.

### 2.4 Protocolo de extração (Sec. 4) — ambiguidades que mudariam os dados

1. **Forma de superfície não especificada.** No Qwen, `run` → id 6108 e ` run` → id 1598: linhas diferentes da matriz. Adotado: forma inicial de palavra (`" " + w`) para Qwen/Gemma.
2. **`[UNK]` passa no critério "exatamente um ID".** Exclusão explícita de unk/especiais + verificação de ida e volta (o id decodifica para a palavra).
3. **"Gemma 2B" é ambíguo** (`gemma-2b`, $d=2048$ × `gemma-2-2b`, $d=2304$). Padrão: `google/gemma-2b`; SHA da revisão registrado.
4. A matriz do Qwen tem linhas de preenchimento (151.936 × 151.665 tokens); o critério de "linha válida" é necessário.

### 2.5 Protocolo do dataset (Sec. 5) — regras que faltavam

O plano não definia: o mapeamento de tags PTB para a partição; o tratamento de auxiliares (*be/have/do*, etiquetados VB* no PTB); nomes próprios (incompatíveis com BERT-uncased); nem os artefatos de tokenização do PTB. Sem uma regra explícita, apareciam `ca` e `wo` (de `ca n't`, `wo n't`) e `de`, `la`, `le`, `del` (de nomes estrangeiros) como "palavras funcionais". Todas essas regras foram fixadas e registradas no JSON. Também falta a fonte: o PTB completo é licenciado pela LDC; usa-se a amostra gratuita do NLTK.

### 2.6 Bugs científicos no código pré-existente (corrigidos)

1. **Geometria falsa na visualização.** `getProjectedCoords` aplicava deformações cosméticas por algoritmo antes de desenhar: rotação e esticamento no PCA, potência 0,8 no t-SNE e torção espiral no Isomap. A nuvem exibida **não era** a saída do algoritmo. Removido; agora há só uma transformação de similaridade isotrópica (preserva razões de distância).
2. **Dados fabricados.** Vetores aleatórios (`np.random`) eram servidos rotulados com o nome do Qwen enquanto o modelo carregava. Se o backend caísse, o frontend exibia em silêncio um JSON estático pré-computado (`semanticSpace.ts`). Tudo removido; sem dados, a UI mostra erro.
3. `CLUSTER_MAP` com 36 palavras escolhidas à mão (removido — Fase 0).
4. `pip install` em tempo de execução, sem versões fixas (removido; `requirements.txt` fixado).
5. Os testes de contrato da API nunca rodavam: o arquivo era `tests/api.py`, que o pytest não coleta, e os scripts apontavam para `test_api.py`. Renomeado.
6. Percurso de caminho no fallback de arquivos estáticos (`/{path}` aceitava `..`) — corrigido.

### 2.7 Desempenho (Sec. 8) — o gargalo provável não era o previsto

Além das $N$ chamadas de desenho, o laço de animação fazia `labelElementsRef.find(...)` dentro de `forEach` sobre os tokens: **$O(N^2)$ por quadro**, mais um *raycast* de oclusão por rótulo. Correções: uma única `BufferGeometry` (RGBA por vértice), índice $O(1)$ de rótulos e *culling* dos 120 rótulos mais próximos. Medição (Edge, 1600×950, $N=438$): ~3–4,6 ms por quadro, laço do retículo 0,14–0,35 ms, 1 chamada de desenho para a nuvem (as outras ~62 são grade e eixos fixos, independentes de $N$). Sprites de canvas ficam para depois, condicionados ao profiling (item em aberto do plano).

---

## 3. Validação em código

| Suíte | Resultado |
|---|---|
| `tests/test_coranking.py` (offline) | 28/28 |
| `tests/test_dataset.py`, `tests/test_models.py` (offline) | 8/8 |
| idem, com rede/cache HF (`-m network`): tokenizers, safetensors = `get_input_embeddings()` bit a bit (Qwen, BERT) | 5/5 |
| `tests/test_api.py` (servidor real) | 46/46 |

---

## 4. Resultado provisório — **não é resultado do TCC**

Dataset provisório (Qwen + BERT; **sem Gemma**), $N=438$ (146 por categoria), seed 42, hash `a4eed31405ab`. Arquivo: `results/main_provisional_a4eed31405ab.json`.

| Modelo | Alg. | $M_T(10)$ | $M_C(10)$ | $Q_{NX}(10)$ | $B_{NX}(10)$ | $B_{TC}(10)$ | Silhouette proj. |
|---|---|---|---|---|---|---|---|
| Qwen | PCA | 0,830 | 0,828 | 0,205 | −0,007 | −0,001 | 0,161 |
| Qwen | t-SNE | 0,878 | 0,896 | 0,483 | +0,017 | +0,018 | 0,257 |
| Qwen | UMAP | 0,906 | 0,922 | 0,491 | +0,019 | +0,016 | 0,362 |
| Qwen | Isomap | 0,792 | 0,910 | 0,248 | +0,028 | +0,117 | 0,281 |
| BERT | PCA | 0,819 | 0,810 | 0,182 | −0,005 | −0,008 | 0,159 |
| BERT | t-SNE | 0,877 | 0,856 | 0,427 | −0,002 | −0,022 | 0,199 |
| BERT | UMAP | 0,895 | 0,904 | 0,427 | +0,011 | +0,009 | 0,256 |
| BERT | Isomap | 0,794 | 0,860 | 0,201 | +0,012 | +0,066 | 0,166 |

Observações a confirmar com o dataset congelado:

- **Separação das classes inflada pela projeção.** O Silhouette no espaço original (cosseno) é de apenas 0,031 (Qwen) e 0,025 (BERT); todas as projeções o elevam para 0,16–0,36. Isso é evidência direta da limitação já anotada no escopo (o Silhouette projetado não distingue agrupamentos genuínos de agrupamentos produzidos pela projeção).
- **$B_{NX}$ e $B_{TC}$ podem divergir em sinal** quando ambos são pequenos (em $K=5$: Qwen/PCA −0,003 × +0,007; BERT/t-SNE +0,002 × −0,032). Eles medem coisas diferentes: $B_{NX}$ só conta eventos leves, sem peso; $B_{TC}$ conta eventos severos, ponderados pelo posto. Reportar ambos.
- Isomap: maior assimetria $M_C \gg M_T$ (comportamento intrusivo forte segundo $B_{TC}$); grafo $k$-NN conexo nos dois modelos.
- PCA: 3 componentes explicam só 7,8% (Qwen) e 11,0% (BERT) da variância.
- **t-SNE com perplexidade 5 ejeta pontos.** Exemplo: BERT, `disclosed` em raio 266 contra p99 = 38. O ponto **não** é isolado em $\mathbb{R}^n$ (cosseno máximo 0,614 > mediana 0,579); o ponto ejetado muda com o seed e ocorre também com `method="exact"`. É um artefato de otimização — mais um motivo para rever a perplexidade antes da coleta.
- $B_{NX,i}(10)$ local: p5–p95 ≈ ±0,2, extremos até ±0,7, em passos de $1/K$. A escala fixa $[-1,1]$ é legível.

---

## 5. Decisões que dependem do autor (não tomadas por esta revisão)

1. Habilitar o Gemma (licença HF + `HF_TOKEN`) e congelar `data/dataset.json` com os três tokenizers.
2. `google/gemma-2b` ou `google/gemma-2-2b`.
3. Perplexidade do t-SNE para $N\approx440$ (manter 5 com nova justificativa, ou rever) — antes da coleta.
4. Política do Isomap para grafo desconexo (ligar × filtrar) e Silhouette em 2D × 3D, alinhando plano e escopo.
5. Política de balanceamento: as mais frequentes (adotada; embeddings melhor treinados) ou amostragem aleatória com seed.
