# ENSy v2.1.3 — Revisão científica e validação da implementação

**Escopo:** revisão do plano `ENSy_v2.1.2_plano_implementacao.md` contra as fontes primárias e contra o código, seguida da implementação de referência (`ensy/`, `app.py`, frontend) e da validação em teste. Data: 2026-10-01.

**Fontes consultadas:** LEE; VERLEYSEN (2009, *Neurocomputing*) e (2008, AISTATS/FSDM) — textos integrais; documento de escopo do TCC; e as fontes do caderno NotebookLM "Matemática para Machine Learning" (74 fontes), lidas pelo CLI `nlm` como **texto bruto** (`nlm source content`, sem resumo por IA) e verificadas por busca literal: VENNA; KASKI (2006, *Neural Networks* e ESANN), VAN DER MAATEN; HINTON (2008), TENENBAUM et al. (2000), MCINNES et al. (2018). Atualizado em 2026-10-02.

---

## 1. Resultado da revisão — o que estava certo

| Item | Verificação |
|---|---|
| Definição de $Q$, blocos $UL/UR/LL/LR$, $D/LT/UT$ | Confere com Lee & Verleysen (2008), Sec. 2 |
| $M_T(K)$, $M_C(K)$ e $G_K$ | Idênticos a `sklearn.manifold.trustworthiness` (erro $<10^{-12}$, $K\in\{5,10,15,20\}$) |
| Nº de K-intrusões severas = nº de K-extrusões severas | Confere com Eq. (8) de 2008; testado |
| $Q_{NX} = U_P+U_N+U_X = \text{LCMC} + K/(N-1)$ | Confere com Eq. (13) de 2008; testado contra a definição direta |
| $B_{NX}(K)=U_X-U_N$ da v2.1.2, positivo = intrusivo | **Confirmado no texto de 2009 (Eq. 22–23) e empiricamente** (ver 2.1) |
| Decomposição local, $\text{mean}_i B_{NX,i} = B_{NX}$ | Exata por construção; testada (erro $<10^{-12}$) |
| Identidade $|F_K(i)| = |M_K(i)|$ (Sec. 9.4) | Testada |
| $\|\hat u-\hat v\|^2 = 2(1-\cos\theta)$ | Correto; consequência: após L2, postos por cosseno = postos euclidianos (testado) |

## 2. Erros e lacunas encontrados no plano

### 2.1 $B_{NX}$: convenção da v2.1.2 confirmada na fonte de 2009

Texto de LEE; VERLEYSEN (2009, p. 1434), literal:

- Eq. 22: $U_N(K)=\frac{1}{KN}\sum_{(k,l)\in LT_K} q_{kl}$ e $U_X(K)=\frac{1}{KN}\sum_{(k,l)\in UT_K} q_{kl}$;
- Eq. 23: $B_{NX}(K) = U_X(K) - U_N(K) = W^{0,0}_X - W^{0,0}_N = -B^{0,0}_{wNX}(K)$, *"Its value is positive for intrusive embeddings and negative for extrusive ones."*

A v2.1.2 citou isso corretamente, e a implementação segue a fonte nos rótulos e no valor. O mesmo trecho confirma a explicação do "paradoxo": uma K-intrusão severa é compensada por uma K-extrusão severa e por $(K - r_{ij})$ mild K-extrusions (Fig. 2) — exatamente o que o teste `test_hard_intrusion_shows_up_as_mild_extrusions` verifica.

**O precursor de 2008 é que é inconsistente:** o texto põe as mild K-intrusions em $LT_K$, mas a Eq. (11) define $U_N$ sobre $UT_K$ e a Eq. (14) define $B_{NX}=U_N-U_X$ — mesmo valor, rótulos trocados — e afirma $B_{NX}=W^{0,0}_N-W^{0,0}_X$, com o sinal oposto. A versão de 2009 corrigiu tudo isso. Recomendação: citar só a versão de 2009 para estas definições.

Âncora empírica independente (mantida nos testes): esfera oca projetada ortograficamente (esmagamento, intrusivo) → $B_{TC}=M_C-M_T>0$ (Eq. 27 de 2009) **e** $B_{NX}>0$; esfera desenrolada (rasgo, extrusivo) → ambos $<0$.

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

### 2.6 Citações verificadas contra o texto das fontes (NotebookLM)

| Afirmação do plano | Texto da fonte | Veredito |
|---|---|---|
| $K$ principal $=10$ (VENNA; KASKI, 2006) | *Neural Networks*, p. 894: *"a human analyst usually looks at a handful (say 10) data points around it. Thus it is very important that the visualization preserves small neighborhoods well"*; os autores avaliam T&C como curvas sobre $k$ (Figs. 1–2) | Apoiada **como heurística**, não como valor derivado. O artigo ESANN 2006, citado no escopo, usa $k=6$ e **não** apoia $K=10$ — citar o de *Neural Networks*, com página |
| Perplexidade 5 | VAN DER MAATEN; HINTON (2008, p. 2582): *"typical values are between 5 and 50"* | Dentro da faixa, no extremo inferior; usar essa citação na justificativa |
| Isomap e grafo desconexo | TENENBAUM et al. (2000, nota 19): pontos desconectados da componente gigante *"can be deleted from further analysis"* | Apoia a **filtragem** prevista no escopo, não a ligação automática do scikit-learn |
| UMAP `n_neighbors=15`, `min_dist=0.1` | O artigo (Sec. 4.3) define $n$ e *min-dist* como hiperparâmetros, sem valores padrão | São defaults da biblioteca `umap-learn` (como o plano já diz); citar a versão da biblioteca |
| $B_{TC}=M_C-M_T$; $Q_{TC}=(M_T+M_C)/2$ | LEE; VERLEYSEN (2009), Eq. 26–27 | Confere (o precursor de 2008 não tinha o fator 1/2) |
| Normalização $C^{0,0}_K = NK$ | LEE; VERLEYSEN (2009), nota 4: o valor exato seria $N\min\{N-2,K\}$; $NK$ é aproximação adotada pelos autores | Sem impacto em $B_{NX}$ (que usa $1/KN$ por definição); registrar se $W^{0,0}$ for reportado |

### 2.7 Bugs científicos no código pré-existente (corrigidos)

1. **Geometria falsa na visualização.** `getProjectedCoords` aplicava deformações cosméticas por algoritmo antes de desenhar: rotação e esticamento no PCA, potência 0,8 no t-SNE e torção espiral no Isomap. A nuvem exibida **não era** a saída do algoritmo. Removido; agora há só uma transformação de similaridade isotrópica (preserva razões de distância).
2. **Dados fabricados.** Vetores aleatórios (`np.random`) eram servidos rotulados com o nome do Qwen enquanto o modelo carregava. Se o backend caísse, o frontend exibia em silêncio um JSON estático pré-computado (`semanticSpace.ts`). Tudo removido; sem dados, a UI mostra erro.
3. `CLUSTER_MAP` com 36 palavras escolhidas à mão (removido — Fase 0).
4. `pip install` em tempo de execução, sem versões fixas (removido; `requirements.txt` fixado).
5. Os testes de contrato da API nunca rodavam: o arquivo era `tests/api.py`, que o pytest não coleta, e os scripts apontavam para `test_api.py`. Renomeado.
6. Percurso de caminho no fallback de arquivos estáticos (`/{path}` aceitava `..`) — corrigido.

### 2.8 Desempenho (Sec. 8) — o gargalo provável não era o previsto

Além das $N$ chamadas de desenho, o laço de animação fazia `labelElementsRef.find(...)` dentro de `forEach` sobre os tokens: **$O(N^2)$ por quadro**, mais um *raycast* de oclusão por rótulo. Correções: uma única `BufferGeometry` (RGBA por vértice), índice $O(1)$ de rótulos e *culling* dos 120 rótulos mais próximos. Medição (Edge, 1600×950, $N=438$): ~3–4,6 ms por quadro, laço do retículo 0,14–0,35 ms, 1 chamada de desenho para a nuvem (as outras ~62 são grade e eixos fixos, independentes de $N$). Sprites de canvas ficam para depois, condicionados ao profiling (item em aberto do plano).

---

## 3. Validação em código

| Suíte | Resultado |
|---|---|
| `tests/test_coranking.py` (offline) | 28/28 |
| `tests/test_dataset.py`, `tests/test_models.py` (offline; inclui decodificação BF16 exata e a política do Isomap) | 10/10 |
| idem, com rede/cache HF (`-m network`): tokenizers; safetensors = `get_input_embeddings()` bit a bit; extração por HTTP Range = download completo bit a bit (Qwen, BERT) | 7/7 |
| `tests/test_api.py` (servidor real) | 46/46 |

---

## 4. Resultados reais (dataset congelado)

Dataset congelado com os quatro tokenizers (Qwen 2.5 0.5B-Instruct, Gemma 2B v1, Gemma 2 2B, BERT base uncased): $N=438$ (146 por categoria), hash `40863cdefcb4`, revisões fixadas no JSON. Extração por HTTP Range (só as 438 linhas de cada matriz). Seed 42. Arquivos: `results/main_40863cdefcb4.json`, `results/ablation_40863cdefcb4.json`, `results/sensitivity_40863cdefcb4.json`.

**Atualização (2026-10-02):** a tabela abaixo é da execução local (Windows) com quatro algoritmos. Os números canônicos — cinco configurações por modelo (t-SNE com perplexidade 5 e 30), Silhouette 2D e 3D, gerados no GitHub Actions a partir de uma árvore limpa (commit `35fffe0`) — estão em `results/` e na Seção 5 de `docs/DOCS.md`. PCA, PCA-NC e Isomap são idênticos entre as plataformas; t-SNE e UMAP diferem entre Windows e Linux mesmo com seed fixo (diferenças da ordem da variação entre seeds), por isso os resultados estocásticos devem ser lidos pela análise de sensibilidade.

### 4.1 Matriz principal, $K=10$

| Modelo | $d$ | Alg. | $M_T$ | $M_C$ | $Q_{NX}$ | $B_{NX}$ | $B_{TC}$ | Silhouette proj. |
|---|---|---|---|---|---|---|---|---|
| Qwen | 896 | PCA | 0,830 | 0,828 | 0,205 | −0,007 | −0,001 | 0,161 |
| Qwen | | t-SNE | 0,878 | 0,896 | 0,483 | +0,017 | +0,018 | 0,257 |
| Qwen | | UMAP | 0,906 | 0,922 | 0,491 | +0,019 | +0,016 | 0,362 |
| Qwen | | Isomap | 0,792 | 0,910 | 0,248 | +0,028 | +0,117 | 0,281 |
| Gemma 1 | 2048 | PCA | 0,847 | 0,875 | 0,222 | +0,002 | +0,028 | 0,307 |
| Gemma 1 | | t-SNE | 0,840 | 0,821 | 0,376 | +0,006 | −0,019 | 0,246 |
| Gemma 1 | | UMAP | 0,875 | 0,921 | 0,397 | +0,017 | +0,046 | 0,399 |
| Gemma 1 | | Isomap | 0,816 | 0,885 | 0,207 | −0,001 | +0,069 | 0,279 |
| Gemma 2 | 2304 | PCA | 0,867 | 0,802 | 0,189 | −0,010 | −0,065 | 0,240 |
| Gemma 2 | | t-SNE | 0,880 | 0,906 | 0,434 | +0,022 | +0,026 | 0,247 |
| Gemma 2 | | UMAP | 0,911 | 0,934 | 0,431 | +0,009 | +0,023 | 0,353 |
| Gemma 2 | | Isomap | 0,864 | 0,920 | 0,238 | +0,012 | +0,057 | 0,298 |
| BERT | 768 | PCA | 0,819 | 0,810 | 0,182 | −0,005 | −0,008 | 0,159 |
| BERT | | t-SNE | 0,877 | 0,856 | 0,427 | −0,002 | −0,022 | 0,199 |
| BERT | | UMAP | 0,895 | 0,904 | 0,427 | +0,011 | +0,009 | 0,256 |
| BERT | | Isomap | 0,794 | 0,860 | 0,201 | +0,012 | +0,066 | 0,166 |

Silhouette no espaço original (cosseno): Qwen 0,031; Gemma 1 0,055; Gemma 2 0,044; BERT 0,025. Variância explicada pelas 3 componentes do PCA: 7,8% (Qwen), 12,6% (Gemma 1), 10,9% (Gemma 2), 11,0% (BERT). O grafo $k$-NN do Isomap foi conexo nos quatro modelos: a política de filtragem não removeu nenhum ponto.

### 4.2 Sensibilidade (5 seeds: 0, 1, 2, 3, 42), $K=10$, média ± DP

| Modelo | t-SNE, perpl. 5: $M_T$ / $B_{NX}$ | t-SNE, perpl. 30: $M_T$ / $B_{NX}$ | UMAP: $M_T$ / $B_{NX}$ |
|---|---|---|---|
| Qwen | 0,880 ± 0,003 / +0,014 ± 0,002 | 0,892 ± 0,006 / **−0,036 ± 0,005** | 0,903 ± 0,003 / +0,019 ± 0,006 |
| Gemma 1 | 0,845 ± 0,004 / +0,012 ± 0,005 | 0,849 ± 0,007 / **−0,036 ± 0,005** | 0,874 ± 0,002 / +0,014 ± 0,003 |
| Gemma 2 | 0,882 ± 0,002 / +0,020 ± 0,006 | 0,893 ± 0,002 / **−0,047 ± 0,007** | 0,914 ± 0,002 / +0,009 ± 0,009 |
| BERT | 0,877 ± 0,002 / −0,001 ± 0,002 | 0,872 ± 0,002 / **−0,039 ± 0,004** | 0,892 ± 0,003 / +0,018 ± 0,007 |

### 4.3 Leituras (descritivas; sem teste de hipótese)

- **O sinal de $B_{NX}$ do t-SNE é determinado pela perplexidade, não pelo modelo.** Com perplexidade 5 o t-SNE é levemente intrusivo (ou neutro, no BERT); com 30 é extrusivo nos quatro modelos, com diferença muito maior que a variação entre seeds. Qualquer afirmação do tipo "o t-SNE é intrusivo/extrusivo" precisa vir acompanhada da perplexidade — e o valor congelado (5) passa a ser uma escolha que decide o resultado, não um detalhe.
- **$M_T$ é robusto ao seed** (DP ≤ 0,007 em todas as configurações); o sinal de $B_{NX}$ do UMAP no Gemma 2 não é (+0,009 ± 0,009).
- **Separação das classes inflada pela projeção:** o Silhouette no espaço original fica entre 0,025 e 0,055; todas as projeções o elevam para 0,16–0,40.
- **Gemma 1 × Gemma 2:** com o mesmo vocabulário (mesmos ids para as 438 palavras), as projeções do Gemma 2 preservam melhor as vizinhanças ($M_T$ maior nos quatro algoritmos); o Gemma 1 tem maior $M_C$ no PCA (0,875 × 0,802).
- **$B_{NX}$ e $B_{TC}$ podem divergir em sinal** quando pequenos (ex.: Gemma 1/t-SNE +0,006 × −0,019): $B_{NX}$ conta só eventos leves, sem peso; $B_{TC}$ conta eventos severos, ponderados pelo posto. Reportar ambos.
- **t-SNE com perplexidade 5 ejeta pontos** de forma dependente do seed (ex.: BERT, `disclosed`, raio 266 × p99 38; o ponto não é isolado em $\mathbb{R}^n$). Artefato de otimização.
- $B_{NX,i}(10)$ local: p5–p95 ≈ ±0,2, extremos até ±0,7, em passos de $1/K$; a escala fixa $[-1,1]$ é legível.

---

## 5. Decisões que dependem do autor

1. **Variante do Gemma:** ambas foram executadas com o protocolo idêntico. Recomendação: reportar as duas (a comparação entre gerações do mesmo modelo é, em si, informativa) e **não** escolher uma a posteriori pelo desempenho — isso seria ajustar o desenho ao resultado.
2. **Perplexidade do t-SNE:** dado o resultado 4.2, recomenda-se reportar o t-SNE nas duas perplexidades (5, congelada, e 30, default da biblioteca) e discutir o efeito, em vez de defender um valor único.
3. Silhouette em 2D × 3D: alinhar plano (3D) e escopo (2D).
4. Política de balanceamento: as mais frequentes (adotada) ou amostragem aleatória com seed.
