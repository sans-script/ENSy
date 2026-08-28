# ENSy v2.0 — Plano de Desenvolvimento Consolidado
## Embedding Navigation System for Neural Language Models

**Instrumento de pesquisa do TCC — UFMA, 2026**

**Status:** v2.1.2 — corrigida a convenção de sinal de $B_{NX}(K)$ contra a fonte primária (LEE; VERLEYSEN, 2009), que havia sido invertida por engano na v2.1.1. Demais itens da v2.1.1 permanecem válidos. Em aberto: execução real do pipeline de dataset e validação empírica de $B_{NX,i}(K)$ com casos sintéticos.

**Escopo:** ferramenta de laboratório local (browser + FastAPI local), sem distribuição, sem desktop shell.

---

## 1. Objetivo

Investigar como diferentes algoritmos de redução de dimensionalidade transformam a geometria de espaços de embedding de modelos de linguagem, e viabilizar a inspeção interativa dessas transformações — não apenas "visualizar embeddings", mas caracterizar quantitativamente a distorção introduzida por cada método, em cada arquitetura de modelo.

O ENSy é o instrumento que produz a evidência do TCC. Ele não é o objeto de estudo em si.

---

## 2. Decisões já congeladas

| Decisão | Estado |
|---|---|
| Arquitetura cliente-servidor local (React/Three.js + FastAPI local) | Congelado |
| Sem Tauri, sem empacotamento desktop, sem distribuição | Congelado — não-objetivo explícito |
| Três modelos: Qwen 2.5 0.5B-Instruct, Gemma 2B, BERT base uncased | Congelado |
| Quatro algoritmos: PCA, t-SNE, UMAP, Isomap | Congelado |
| Framework de co-ranking (Lee & Verleysen, 2009) para métricas de qualidade | Congelado |
| Dataset baseado em Penn Treebank com partição de referência linguística | Congelado (regra de tamanho — ver Seção 5) |
| Normalização L2 obrigatória antes de qualquer métrica | Congelado |
| Cache em memória (sem L1/L2/L3 em disco) | Congelado |

---

## 3. Não-objetivos (reafirmados)

- Distribuição como aplicativo desktop ou instalador (Tauri, PyInstaller, etc.)
- Suporte a modelos via API remota (ChatGPT, Gemini API) — quebra reprodutibilidade
- Edge bundling tridimensional
- FAISS / HNSW para $N > 1000$
- Interface para usuários externos ao laboratório
- Framework genérico de runtime local-first
- Análise de anisotropia em tempo real (F4 — roadmap pós-graduação)
- Renderização otimizada para $N \gg 1000$ (F7 — roadmap pós-graduação)

**Nota sobre performance:** o problema de FPS relatado (queda de frame rate com $N$ grande) não é um problema de navegador vs. desktop — o WebView do Tauri usa o mesmo motor WebGL do sistema (WebView2/WebKit), então uma migração para desktop não resolveria isso. O problema é de implementação de renderização (Seção 8) e é resolvido inteiramente dentro do navegador.

---

## 4. Protocolo de extração de embeddings

**Decisão de design:** para o experimento principal, o ENSy extrai exclusivamente a matriz de embedding de entrada estática de cada modelo — `model.get_input_embeddings().weight` para Qwen/Gemma (decoder-only) e `bert.embeddings.word_embeddings.weight` para BERT (encoder-only).

**Justificativa:** é uma representação diretamente comparável entre as três arquiteturas, pois corresponde à matriz de embeddings de entrada antes da aplicação das camadas contextuais específicas de cada arquitetura. Extrair hidden states contextualizados exigiria decidir camada + contexto + pooling, o que muda o que está sendo medido e compromete a comparabilidade entre os três modelos.

**Consequência para o enquadramento do TCC:** o objeto de estudo não é "embeddings de linguagem" em sentido contextual — é especificamente a geometria das matrizes de embeddings de entrada de modelos de linguagem pré-treinados. Essa formulação, mais restrita, é também mais defensável na banca.

### Protocolo de Extração — v2.1

- **Unidade textual:** token/palavra isolada
- **Representação:** matriz de embedding de entrada (pré-encoder/pré-decoder)
- **Contexto:** nenhum (lookup estático, sem forward pass do modelo)
- **Pooling:** não aplicável (um único vetor por token)
- **Critério de inclusão:** uma unidade lexical só entra no conjunto ativo se sua representação pelo tokenizer de **cada um** dos três modelos produzir **exatamente um ID de token**, e esse ID corresponder a uma **linha válida da matriz de embeddings de entrada** daquele modelo. Chama-se isso de "interseção de unidades lexicais com representação single-token" — não "interseção de vocabulário", porque não existe necessariamente um vocabulário compartilhado no sentido de IDs comuns entre os três tokenizers; o que existe é uma interseção de formas lexicais que, cada uma no seu próprio tokenizer, resolve para um único token válido.
- **dtype:** float32
- **Normalização:** L2, aplicada após extração, antes de qualquer métrica

**Nota de precisão sobre "token único":** "a palavra é representada por um único token" não implica "a palavra possui um token lexical independente" — o mesmo string pode ser um único token num tokenizer e uma sequência de subwords em outro. O critério de inclusão acima é operacionalmente inequívoco (testável por código, não por interpretação).

Hidden states contextualizados ficam como extensão futura (fora do experimento principal do TCC).

**Cuidado com causalidade na comparação entre modelos:** Qwen, Gemma e BERT diferem não só em arquitetura (decoder-only vs. encoder-only), mas também em dimensão, vocabulário, tokenizador, objetivo de pré-treinamento, corpus e receita de treinamento. O experimento não isola qual desses fatores produz um padrão de distorção observado. Redação adotada: "permite observar se os padrões de distorção dos algoritmos se mantêm ou variam sob diferentes modelos e arquiteturas, sem atribuir causalidade exclusivamente à arquitetura."

**Ação pendente:** Gemma 2B é um modelo gated no Hugging Face — requer aceitar licença e autenticação por token antes do download programático. Resolver na Fase 0, antes de qualquer código de abstração de modelo.

---

## 5. Protocolo de dataset lexical derivado do Penn Treebank

**Nome:** "Dataset lexical derivado do Penn Treebank" — não "dataset baseado no PTB", porque não se usa o PTB diretamente como matriz de exemplos (que é contextual, anotado por ocorrência). Constrói-se um subconjunto lexical curado (tipos, não ocorrências): PTB tokens → lexical types.

**Taxonomia:** `content_noun` / `content_verb` / `function_word`, chamada de "partição de referência linguística" — não "classes gramaticais". "Stopword" é categoria funcional/lexical por convenção, não uma classe gramatical no mesmo nível de "substantivo" e "verbo"; a taxonomia corrigida evita crítica linguística na banca.

```
Penn Treebank
      |
normalização (case, pontuação)
      |
vocabulário candidato com partição de referência
(content_noun/content_verb/function_word)
      |
tokenização independente em Qwen, Gemma, BERT
      |
interseção: token deve ser single-token nos três
vocabulários (critério da Seção 4)
      |
deduplicação (ver regras abaixo)
      |
dataset final (N = o que sobrar, distribuição
balanceada entre categorias)
```

**Regras de deduplicação e desambiguação:**

- Cada unidade lexical aparece uma única vez no dataset
- Tokens com a mesma forma normalizada são deduplicados
- **Regra de desambiguação de homógrafos/polissemia:** formas lexicais que aparecem no PTB com mais de uma categoria (ex.: *run* como verbo e substantivo) recebem a partição de referência segundo uma regra de seleção fixa, definida previamente e aplicada uniformemente a todos os itens (categoria de maior frequência de ocorrência no corpus). Formas cuja categoria não possa ser determinada de forma não ambígua por essa regra são **excluídas** do dataset — não incluídas com uma categoria escolhida ad-hoc
- O rótulo atribuído representa a categoria escolhida no dataset curado, não uma inferência do modelo

**Critério de aceitação:** "o dataset deve maximizar itens no vocabulário comum aos três modelos, mantendo distribuição equilibrada entre categorias de referência" — não um número fixo. Se resultar $N=137$, o TCC reporta $N=137$; não infla artificialmente para 150.

O dataset final é registrado como arquivo JSON versionado no repositório (reprodutibilidade).

**Sequência operacional completa (21 passos, executar antes do pipeline de métricas; engenharia de renderização/infraestrutura pode avançar em paralelo):**

```
1. Qwen tokenizer          8. deduplicação
2. Gemma tokenizer         9. balanceamento
3. BERT tokenizer         10. N final -> congelar dataset.json + hash
4. PTB -> lexical types    11. extrair embeddings
5. classificação           12. verificar dimensões
   linguística             13. normalização L2
6. desambiguação de        14. matriz de distâncias
   homógrafos              15. co-ranking Q
7. single-token filtering  16. métricas
                               |
                     17. PCA / 18. t-SNE / 19. UMAP / 20. Isomap
                               |
                     21. as 12 configurações
```

O dataset (`dataset.json` + hash, ao final do passo 10) é congelado antes de qualquer extração de embedding ou métrica — vira uma entrada experimental fixa, em vez de algo que muda enquanto o resto do sistema é desenvolvido.

---

## 6. Pré-processamento — regras e nomenclatura

| Item | Regra |
|---|---|
| **Normalização L2** | Obrigatória, sempre. Não elimina anisotropia — apenas torna a geometria do experimento equivalente à similaridade de cosseno ($\|\hat u-\hat v\|^2=2(1-\cos\theta)$). Não afirmar que "corrige" ou "resolve" anisotropia. |
| **Class-Conditional Centering** (ex-"center-shifting") | Renomeado para deixar claro que usa rótulos. Modo exploratório opcional, desativado por padrão, com aviso na UI sobre o viés introduzido. Não participa dos benchmarks quantitativos nem dos resultados principais do TCC. |
| **PCA — centered** | Versão principal, com centralização ($X_c = X-\mu$), implementada via SVD direto da matriz centralizada ($X_c = U\Sigma V^\top$, $Y = X_c V_{1:3}$) — evitar construir explicitamente a matriz de covariância. |
| **PCA-NC — non-centered** (ablação) | Variante sem centralização, usada apenas como modo de ablação explícito, nunca reportada como "PCA" sem qualificador. |

---

## 7. Parâmetros dos algoritmos — tabela congelada

Nenhum algoritmo entra em produção sem esta tabela preenchida e travada antes da coleta de dados:

| Algoritmo | Parâmetro | Valor |
|---|---|---|
| PCA | n_components | 3 |
| PCA | centering | sim (ablação: PCA-NC) |
| t-SNE | perplexity | 5, fixo — não manter a fórmula dinâmica $\min(5, \max(2, \lfloor N/4\rfloor))$: para $N\geq20$ ela sempre resulta em 5, então é mais limpo declarar diretamente "perplexidade fixa de 5, adotada devido ao pequeno tamanho amostral do experimento" |
| t-SNE | random_state | 42 |
| t-SNE | metric (alta dimensão) | cosine — registrar explicitamente que $D_X = D_{\cos}(X)$ na construção das relações de vizinhança, enquanto a avaliação/visualização usa $D_Y = D_{\text{euclidiana}}(Y)$ no espaço projetado $\mathbb{R}^3$ |
| UMAP | n_neighbors | 15 (default da biblioteca) |
| UMAP | min_dist | 0.1 (default da biblioteca) |
| UMAP | metric | cosine |
| UMAP | random_state | 42 |
| Isomap | n_neighbors | 15 — escala de vizinhança fixa adotada para comparabilidade aproximada com o UMAP, não valor canônico do algoritmo |
| Isomap | metric | cosine |

Para algoritmos estocásticos (t-SNE, UMAP): seed fixo garante reprodutibilidade do resultado reportado, mas não caracteriza estabilidade. Se houver tempo, rodar com múltiplos seeds como análise de sensibilidade separada — não como parte do experimento principal.

**Valor de vantagem metodológica:** usar os defaults documentados do UMAP (em vez de escolher valores que fazem cada algoritmo "ficar bonito" na visualização) protege contra a crítica de que os hiperparâmetros foram ajustados a posteriori para favorecer um resultado.

---

## 8. Arquitetura de renderização (correção de performance)

O gargalo de FPS relatado com $N$ grande tem duas causas prováveis, ambas resolvidas dentro do navegador, sem necessidade de shell nativo:

### 8.1 Consolidação de geometria

Trocar $N$ objetos `THREE.Points` individuais (um por token, um draw call cada) por uma única `BufferGeometry` contendo todas as posições em um único array — um draw call para a nuvem inteira, independente de $N$.

### 8.2 Labels: sair do overlay de DOM

O sistema atual de labels em `<div>` reprojetados via JS a cada frame (padrão CSS2DRenderer) pode se tornar gargalo relevante à medida que $N$ aumenta — hipótese de engenharia plausível dado o $N$ alvo do TCC (150–200 tokens), mas que deve ser confirmada por profiling do navegador (Seção 8.4), não assumida como fato geral.

**Substituição recomendada:** renderizar os labels como sprites de canvas (`THREE.Sprite` + `CanvasTexture`, texto desenhado em textura, billboard voltado à câmera) — elimina o custo de atualização dos elementos de texto no DOM (não é "custo zero": ainda existe custo de textura, upload à GPU, fragment shading e draw calls, só que num regime que escala muito melhor que $N$ elementos de DOM).

Se labels em `<div>` forem mantidos, aplicar culling: atualizar apenas os labels visíveis/próximos da câmera a cada frame, não todos os $N$.

### 8.3 Seleção por reticle

O mecanismo atual (Seção 5.6 do artigo SMVV v1) itera sobre todos os $N$ pontos por frame em JS puro — aceitável até alguns milhares de tokens no $N$ alvo do TCC (150–200). Não requer mudança agora; reavaliar apenas se $N$ crescer além de ~2000 (fora do escopo atual).

### 8.4 Profiling — engenharia, não resultado científico

Como o ENSy é instrumento de pesquisa, o profiling de performance deve ser simples e não entra nos resultados científicos principais do TCC — serve apenas para garantir que a ferramenta funciona:

- FPS / frame time
- CPU time do loop de seleção por reticle
- Número de draw calls
- Memória aproximada de GPU

Isso confirma (ou refuta) empiricamente as hipóteses de gargalo da Seção 8.2, em vez de tratá-las como fato assumido.

---

## 9. Framework de co-ranking e métricas — fundamentação

Fonte primária: **LEE, J. A.; VERLEYSEN, M. Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, v. 72, p. 1431–1443, 2009.**

Implementar na ordem: $Q$ → Trustworthiness/Continuity → $B_{NX}$ → $Q_{NX}$ → Silhouette. Não pular para F2/F3 (câmera térmica, probing) antes de F1 (matriz $Q$) estar funcional e validada.

### 9.0 Notação e definições da matriz de co-ranking

Dados $N$ pontos, $r_{ij}$ é o posto (rank) de $j$ em relação a $i$ no espaço de alta dimensão, e $\hat r_{ij}$ o posto correspondente no espaço projetado. A matriz de co-ranking é:

$$Q = [q_{kl}]_{1 \le k,l \le N-1}, \quad q_{kl} = |\{(i,j) : r_{ij}=k \text{ e } \hat r_{ij}=l\}|$$

Custo computacional: $O(N^2 \log N)$ (2$N$ operações de ordenação).

A matriz $Q$ é dividida em quatro blocos de tamanho $K$: $UL_K$ (upper-left), $UR_K$ (upper-right), $LL_K$ (lower-left), $LR_K$ (lower-right). O bloco $UL_K$ é adicionalmente dividido em diagonal $D_K$, triângulo inferior $LT_K = \{(i,j): 1<i\le K, j<i\}$, e triângulo superior $UT_K = \{(i,j): 1\le i<K, j>i\}$.

**Definição de intrusão e extrusão (LEE; VERLEYSEN, 2009, Seção 2):** o erro de rank é definido como $r_{ij} - \hat r_{ij}$. Uma **intrusão** é um erro de rank positivo para algum par — ou seja, $\hat r_{ij} \le K < r_{ij}$: o ponto $j$ é um intruso na vizinhança $K$-ária projetada de $i$, sem pertencer à vizinhança genuína de alta dimensão. Uma **extrusão** é um erro de rank negativo: $r_{ij} \le K < \hat r_{ij}$, um vizinho genuíno de alta dimensão que foi ejetado da vizinhança projetada.

**K-intrusões leves (mild) e severas (hard):** uma K-intrusão leve corresponde ao evento $\hat r_{ij} < r_{ij} \le K$ — o par permanece dentro de ambas as vizinhanças de tamanho $K$, mas seu rank piorou. Uma K-intrusão severa corresponde a $\hat r_{ij} \le K < r_{ij}$ — o par só está na vizinhança projetada. Definições simétricas valem para K-extrusões leves e severas. As K-intrusões (leves e severas) ocupam os trapézios $LT_K \cup LL_K$; as K-extrusões ocupam $UT_K \cup UR_K$.

### 9.1 Trustworthiness $M_T(K)$ e Continuity $M_C(K)$

$$M_T(K) = 1 - \frac{2}{G_K} \sum_{(k,l) \in LL_K} (k-K)\, q_{kl}$$

$$M_C(K) = 1 - \frac{2}{G_K} \sum_{(k,l) \in UR_K} (l-K)\, q_{kl}$$

$$G_K = \begin{cases} NK(2N-3K-1) & \text{se } K<N/2 \\ N(N-K)(N-K-1) & \text{se } K \ge N/2 \end{cases}$$

Trustworthiness é penalizada por K-intrusões severas (vetores distantes que se tornam vizinhos falsos); Continuity é penalizada por K-extrusões severas (vizinhos reais projetados para longe).

### 9.2 $B_{NX}(K)$ — Indicador de comportamento (convenção corrigida)

**Correção crítica desta versão:** a v2.1.1 do plano adotou a convenção $B_{NX} = U_N - U_X$ (positivo = intrusivo), citando uma suposta verificação contra a implementação R `coRanking`. Essa citação não pôde ser confirmada contra uma fonte primária e está **em conflito direto com o artigo original**. A fonte primária (LEE; VERLEYSEN, 2009, Eq. 22–23, p. 1434–1435) define:

$$U_N(K) = \frac{1}{KN}\sum_{(k,l)\in LT_K} q_{kl} \quad \text{(fração de mild K-intrusions)}$$

$$U_X(K) = \frac{1}{KN}\sum_{(k,l)\in UT_K} q_{kl} \quad \text{(fração de mild K-extrusions)}$$

$$\boxed{B_{NX}(K) = U_X(K) - U_N(K)}$$

*"Its value is positive for intrusive embeddings and negative for extrusive ones."* (LEE; VERLEYSEN, 2009, p. 1435)

**O resultado é contraintuitivo à primeira vista** — $B_{NX}>0$ depende de $U_X$ (mild extrusions) ser maior que $U_N$ (mild intrusions), não o oposto. A explicação está na Seção 4 do artigo: os autores demonstram (Fig. 2, p. 1435) que o número de K-intrusões severas é sempre igual ao número de K-extrusões severas — é uma propriedade estrutural da matriz $Q$ (cada linha e cada coluna soma exatamente $N$). O que os distingue é o efeito colateral: uma única K-intrusão severa desloca vários vizinhos subsequentes, gerando múltiplas K-extrusões leves proporcionais à severidade da intrusão. Ou seja: **a magnitude de $U_X$ (extrusões leves) mede indiretamente a severidade das intrusões severas**, e vice-versa para $U_N$. Por isso o sinal do indicador aponta na direção aparentemente invertida.

**Decisão adotada pelo ENSy a partir desta versão:**

$$B_{NX}(K) = U_X(K) - U_N(K)$$

- $B_{NX}(K) > 0$: comportamento **intrusivo** (crowding — ex.: t-SNE em áreas esparsas)
- $B_{NX}(K) < 0$: comportamento **extrusivo** (tearing — ex.: Isomap sob colapso de componentes)
- $B_{NX}(K) \approx 0$: comportamento aproximadamente equilibrado

Esta é a fórmula do artigo original, sem intermediação de uma implementação de terceiros. Qualquer implementação de referência (biblioteca R, Python etc.) usada futuramente para validação cruzada deve ser conferida contra esta definição — não o contrário.

**Validação obrigatória antes de codificar a câmera térmica**, com casos sintéticos:

1. **Caso A — identidade:** $r_{ij} = \hat r_{ij}$ para todos os pares. Resultado esperado: $B_{NX}(K) \approx 0$.
2. **Caso B — promoção intrusiva:** pares construídos com $\hat r_{ij} < r_{ij} \le K$ (mild K-intrusion, aumenta $U_N$). Resultado esperado: $B_{NX}(K) < 0$ (mais mild intrusions relativas a extrusions).
3. **Caso C — degradação extrusiva:** pares construídos com $r_{ij} < \hat r_{ij} \le K$ (mild K-extrusion, aumenta $U_X$). Resultado esperado: $B_{NX}(K) > 0$.

**Nota importante sobre os testes:** os casos sintéticos devem ser construídos por **permutação controlada de ranks**, não por deslocamento geométrico de pontos — um deslocamento geométrico pode produzir K-intrusões severas em vez de leves, que não contribuem para $U_N$/$U_X$ da forma esperada. Os testes B e C acima já refletem essa correção (permutação direta de rank, não posição espacial).

### 9.3 Preservação Geral $Q_{NX}(K)$

$$Q_{NX}(K) = U_P(K) + U_N(K) + U_X(K) = Q_{LC}(K) + \frac{K}{N-1}$$

onde $U_P(K) = \frac{1}{KN}\sum_{(k,l)\in D_K} q_{kl}$ é a fração de pares que mantêm exatamente o mesmo rank em ambas as vizinhanças, e $Q_{LC}(K)$ é a métrica LCMC (Local Continuity Meta-Criterion). $Q_{NX}(K)$ é mais permissiva que a versão ponderada $Q_{wNX}$, pois só decresce com K-intrusões e K-extrusões severas.

### 9.4 $B_{NX,i}(K)$ — decomposição local para a câmera térmica

O $B_{NX}(K)$ das seções anteriores é uma **estatística global**, agregada sobre todos os pares $(i,j)$. A câmera térmica (F2) precisa de um valor **por token**.

**Uma tentativa ingênua falha:** definir $B_{NX,i}(K)$ a partir dos conjuntos de vizinhos falsos/perdidos do probing (F3) — $F_K(i) = n_K(i)\setminus N_K(i)$ (falsos) e $M_K(i) = N_K(i)\setminus n_K(i)$ (perdidos) — resulta sempre em zero, porque quando $|N_K(i)|=|n_K(i)|=K$, tem-se $|F_K(i)| = K - |T_K(i)| = |M_K(i)|$ por construção. A contagem de falsos e de perdidos é sempre igual ponto a ponto. Essa é exatamente a mesma identidade estrutural (linhas/colunas de $Q$ somam $N$) que gera o paradoxo de sinal da Seção 9.2 — não uma coincidência.

**Definição correta:** decompor a matriz de co-ranking **antes** de agregar sobre todos os pontos. Cada entrada $q_{kl}$ é, por construção, uma soma sobre pares: $q_{kl} = \sum_{i=1}^N q_{kl}^{(i)}$, onde $q_{kl}^{(i)}$ conta, para um $i$ fixo, os $j$ tais que $r_{ij}=k$ e $\hat r_{ij}=l$.

Define-se, aplicando as mesmas máscaras triangulares da Seção 9.2 a $q_{kl}^{(i)}$:

$$U_{N,i}(K) = \frac{1}{K}\sum_{(k,l)\in LT_K} q_{kl}^{(i)}, \qquad U_{X,i}(K) = \frac{1}{K}\sum_{(k,l)\in UT_K} q_{kl}^{(i)}$$

$$\boxed{B_{NX,i}(K) = U_{X,i}(K) - U_{N,i}(K)}$$

consistente com a convenção global corrigida da Seção 9.2 (não a fórmula com sinal trocado usada em versões anteriores deste plano).

**Propriedade de consistência (testável por código):**

$$B_{NX}(K) = \frac{1}{N}\sum_{i=1}^{N} B_{NX,i}(K)$$

Este é o teste automatizado obrigatório antes de liberar a câmera térmica:

$$\left| B_{NX}(K) - \frac{1}{N}\sum_{i=1}^N B_{NX,i}(K) \right| < \epsilon$$

para todos os $K$ testados. Se esse teste passar, há garantia de que a câmera térmica é uma decomposição local matematicamente consistente do $B_{NX}$ global, e não uma métrica inventada separadamente.

**Intervalo do escore local:** com $B_{NX,i}(K) = (U_{X,i}-U_{N,i})/K$, tem-se aproximadamente $-1 \le B_{NX,i}(K) \le 1$. Isso permite uma escala bipolar normalizada e **global e fixa** em $[-1,1]$ para a câmera térmica — não normalizada dinamicamente por frame ou por projeção. Caso contrário, dois algoritmos poderiam ter o mesmo valor local e receber cores diferentes dependendo da distribuição dos demais pontos naquele frame específico.

**Validação obrigatória:** os mesmos três casos sintéticos da Seção 9.2, agora ponto-a-ponto (ex.: no Caso C, os pontos com pares artificialmente afastados em rank devem mostrar $B_{NX,i}>0$ enquanto os demais permanecem próximos de zero).

**Precisão semântica para a UI/documentação:** a câmera térmica não mostra "quanto este token foi distorcido" — mostra "qual é a contribuição líquida deste token para o comportamento intrusivo/extrusivo, segundo as mild intrusions/extrusions em $K$". Um ponto pode sofrer muitas K-intrusões severas e ainda assim ter $B_{NX,i}(K) \approx 0$, porque intrusões severas não entram diretamente nesta estatística local (elas se manifestam indiretamente via $U_{X,i}$, pela mesma relação estrutural da Seção 9.2). Nomenclatura recomendada na UI: **"Câmera térmica — contribuição local para $B_{NX}(K)$"**, não "distorção local".

### 9.5 Silhouette Score com partição de referência linguística

$$s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}$$

Calculado sobre as coordenadas projetadas $Y_i \in \mathbb{R}^3$, usando `content_noun`/`content_verb`/`function_word` como partição de referência (ROUSSEEUW, 1987). Redação adotada: "fornece medida complementar de separação das classes no espaço projetado" — não "distingue crowding" (Silhouette mede separação, não identifica causa).

**Não aplicar Class-Conditional Centering antes deste cálculo** — constitui data leakage: usar rótulos de classe para centralizar antes de um algoritmo de projeção não supervisionado infla artificialmente a separação inter-classe na alta dimensão, invalidando o uso do Silhouette para avaliar a capacidade autônoma da projeção.

### 9.6 Correções de terminologia

| Termo anterior | Termo adotado | Motivo |
|---|---|---|
| "ground truth" gramatical / "classes gramaticais" | "partição de referência linguística" | POS tags não são uma verdade semântica do embedding; "stopword" também não é uma classe gramatical homogênea com noun/verb |
| "Silhouette distingue crowding" | "Fornece medida complementar de separação das classes no espaço projetado" | Silhouette mede separação, não identifica causa |
| "N < 100 = instabilidade estatística" | "Amostragem reduzida — indicadores podem apresentar maior variabilidade" | Threshold de 100 é convenção de engenharia, não propriedade estatística com fonte |
| "$B_{NX}$ verificado contra coRanking (R)" | "$B_{NX}$ definido pela fonte primária (LEE; VERLEYSEN, 2009)" | A citação da implementação R não pôde ser confirmada e estava em conflito com o artigo original |

### 9.7 Sistema de cores — camadas visuais, câmera térmica e probing (F2/F3)

O sistema de cor por cluster semântico fixo do SMVV v1 (hardcoded junto ao `CLUSTER_MAP`) é eliminado junto com o vocabulário fixo (Seção 4). Substituído por três camadas que coexistem sem conflito de canal visual:

**Camada 1 — Cor base (partição de referência linguística).** Estado padrão de repouso. Cada token é colorido pela categoria de referência do dataset (`content_noun`/`content_verb`/`function_word`), substituindo diretamente o campo `color` que hoje vem hardcoded do `token_registry`.

**Camada 2 — Câmera térmica bipolar (F2, toggle global).** Baseada em $B_{NX,i}(K)$ (Seção 9.4), escala global fixa em $[-1,1]$:

| Faixa de $B_{NX,i}(K)$ | Cor | Interpretação |
|---|---|---|
| $< 0$ | Azul/Ciano | Extrusão (tearing) |
| $\approx 0$ | **Cinza claro** (nunca verde) | Alta fidelidade / neutro |
| $> 0$ | Vermelho/Laranja | Intrusão (crowding) |

**Restrição de canal:** cinza claro no ponto neutro, nunca verde — verde é reservado exclusivamente à Camada 3.

**Camada 3 — Probing de vizinhança (F3, ativa só no token selecionado pelo reticle).**

| Cor | Categoria | Definição |
|---|---|---|
| Verde | True Neighbors ($T_K = N_K \cap n_K$) | Vizinhos reais preservados na projeção |
| Vermelho | False Neighbors ($F_K = n_K \setminus N_K$) | Falsos vizinhos introduzidos pela projeção |
| Azul translúcido | Missing Neighbors ($M_K = N_K \setminus n_K$) | Vizinhos reais ejetados da projeção |

**Regra de precedência:** Probing (3) > Câmera térmica (2) > Cor base (1). As camadas nunca se misturam simultaneamente no mesmo ponto — quando o probing está ativo, substitui visualmente a cor daquele token específico.

**Implicação de implementação:** com os pontos vivendo em uma única `BufferGeometry` (Seção 8.1), a cor é um atributo de vértice (`geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3))`, material com `vertexColors: true`). Trocar de camada é reescrever o array de cores e marcar `colorAttribute.needsUpdate = true` — operação $O(N)$, não troca de $N$ materiais.

**Dependência bloqueante:** nenhuma cor da Camada 2 é implementada em código antes de fechar a validação de $B_{NX,i}(K)$ (Seção 9.4) contra os casos sintéticos — implementar com a fórmula errada seria o tipo de erro que só aparece na defesa do TCC.

---

## 10. Protocolo experimental

```
### Variáveis independentes
- modelo (Qwen, Gemma, BERT)
- algoritmo (PCA, t-SNE, UMAP, Isomap)
- seed (fixo=42 para resultado principal; múltiplos seeds
  só em análise de sensibilidade separada)
- K (vizinhança das métricas de co-ranking)

### Protocolo de K
- K principal = 10 (referência: VENNA; KASKI, 2006)
- K de análise = {5, 10, 15, 20} — calcular as quatro
  métricas ao longo dessa faixa e apresentar como curvas,
  não como valor único

### Variáveis dependentes
- Trustworthiness M_T(K)
- Continuity M_C(K)
- B_NX(K)
- Q_NX(K)
- Silhouette (com partição de referência linguística)

### Constantes (travadas antes da coleta)
- dataset (arquivo JSON versionado)
- protocolo de extração de embedding (Seção 4)
- normalização (L2)
- métrica de distância (cosine)
- parâmetros de cada algoritmo (Seção 7)
- hardware e versões de biblioteca

### Reprodutibilidade
- random_seed = 42
- commit_sha do código
- versões: Python, PyTorch, NumPy, UMAP, scikit-learn
- dataset_hash
```

Apresentar como curvas sobre $K$ (não como número único) deixa o resultado do TCC mais robusto e alinhado com o próprio framework de co-ranking, que é intrinsecamente dependente de escala.

Matriz de experimentos principal: $3 \text{ modelos} \times 4 \text{ algoritmos} = 12$ configurações. Qualquer variação adicional (PCA-NC, class-conditional centering, múltiplos seeds) é **ablação/sensibilidade**, documentada e reportada separadamente — nunca misturada com os 12 resultados principais.

**Estrutura de apresentação dos resultados** (cada configuração produz um conjunto de curvas):

```
Modelo
 |-- Algoritmo (PCA / t-SNE / UMAP / Isomap)
      |-- Trustworthiness(K) -- curva sobre K={5,10,15,20}
      |-- Continuity(K)      -- curva sobre K={5,10,15,20}
      |-- B_NX(K)            -- curva sobre K={5,10,15,20}
      |-- Q_NX(K)            -- curva sobre K={5,10,15,20}
      +-- Silhouette         -- valor único (não depende de K)
```

**Cache key** (evita recarregar modelo a cada troca de projeção):

```
cache_key = (model_id, dataset_hash, extraction_config, preprocessing_config)
```

---

## 11. Ordem de implementação

| Fase | Entregável | Bloqueia |
|---|---|---|
| **0** | Remover `CLUSTER_MAP` hardcoded e bootstrap vectors aleatórios | Validade científica |
| **0** | Resolver acesso gated ao Gemma 2B (licença HF) | Fase de modelos |
| **0** | Endpoint `POST /api/dataset` + pipeline de interseção de vocabulário (Seção 5) | Dataset real |
| **0** | Camada de abstração de modelos (Qwen, Gemma, BERT) com protocolo de extração fixo (Seção 4) | Experimentos comparativos |
| **0** | Cache de embeddings em memória | Performance durante experimentos |
| **0** | Correção de renderização: `BufferGeometry` única + sprites de canvas para labels (Seção 8) | Uso prático em $N=150$–200 sem travar |
| **0** | Alerta de amostragem reduzida na UI, linguagem corrigida (Seção 9.6) | Honestidade científica |
| **1** | Normalização L2 + Matriz de co-ranking $Q$ + Trustworthiness/Continuity (Seção 9.0–9.1) | F2, F3 |
| **1** | Tabela de parâmetros travada (Seção 7) antes de rodar qualquer coleta | Validade experimental |
| **1** | Validação sintética de $B_{NX}(K)$ com a convenção corrigida (Seção 9.2) | F2 |
| **1** | Definição e validação sintética de $B_{NX,i}(K)$ com teste de consistência (Seção 9.4) | F2 |
| **2** | F2 — Câmera térmica (desbloqueada só após validação da Seção 9.4) | — |
| **2** | F3 — Probing colorido de vizinhança | — |
| **3** | F6 — Silhouette com partição de referência linguística (depende só de F1) | — |
| **3** | F5 — Painel comparativo (12 configurações × curvas de K; pode exibir F6, mas não depende dele) | — |

**Nota de dependência:** F6 (Silhouette) depende apenas da projeção + rótulos, ou seja, de F1 — não de F5. A ordem correta é F1 → {F2, F3, F5, F6} em paralelo (F2 com o bloqueio adicional da validação de $B_{NX,i}$), não uma cadeia sequencial estrita.

---

## 12. Critérios de sucesso do MVP de pesquisa

- [ ] Dataset lexical derivado do PTB, deduplicado, com partição de referência linguística e $N$ reportado honestamente (não forçado a 150)
- [ ] Protocolo de extração de embedding documentado e idêntico entre os 3 modelos, com critério de inclusão operacionalmente inequívoco
- [ ] Tabela de parâmetros dos 4 algoritmos travada antes da coleta
- [ ] $B_{NX}(K) = U_X(K) - U_N(K)$, conforme a fonte primária, validado contra casos sintéticos antes de qualquer visualização térmica
- [ ] $B_{NX,i}(K)$ definido, implementado e validado (incluindo o teste de consistência $B_{NX}(K) = \text{mean}_i\, B_{NX,i}(K)$) antes da implementação da câmera térmica
- [ ] 12 configurações (3 modelos × 4 algoritmos) rodadas com seed fixo, métricas calculadas como curvas sobre $K=\{5,10,15,20\}$, resultados versionados
- [ ] Renderização interativa em $N=150$–200 com desempenho aceitável segundo métricas de profiling previamente definidas (frame time, FPS, draw calls, tempo do loop de seleção — Seção 8.4), com thresholds numéricos fixados somente após medição em hardware/navegador real
- [ ] Terminologia corrigida (partição de referência linguística, PCA centered/non-centered, class-conditional centering, causalidade de arquitetura suavizada) em todo o documento final

---

## 13. Itens em aberto — resolver antes de escrever código

### Resolvidos

1. ~~Hiperparâmetros de UMAP/Isomap/t-SNE~~ → travados na Seção 7
2. ~~Valor de K para as métricas de co-ranking~~ → K principal=10, K de análise={5,10,15,20}, Seção 10
3. ~~Taxonomia linguística~~ → `content_noun/content_verb/function_word`, Seção 5
4. ~~Regra de desambiguação de homógrafos~~ → regra de seleção fixa por frequência + exclusão de casos ambíguos, Seção 5
5. ~~Precisão do critério single-token~~ → "interseção de unidades lexicais com representação single-token", Seção 4
6. ~~Inconsistência noun/verb/stopword na Camada 1 de cores~~ → corrigido, Seção 9.7
7. ~~Critério de sucesso "sem queda de FPS"~~ → substituído por critério mensurável via profiling, Seção 12
8. ~~Convenção de sinal de $B_{NX}(K)$~~ → corrigida contra a fonte primária (LEE; VERLEYSEN, 2009): $B_{NX} = U_X - U_N$, positivo = intrusivo. **A versão anterior deste plano (v2.1.1) tinha o sinal invertido** e citava uma verificação contra implementação de terceiros que não pôde ser confirmada.
9. ~~Definição matemática de $B_{NX,i}(K)$~~ → decomposição da matriz de co-ranking por ponto antes da agregação, Seção 9.4, com convenção de sinal consistente com o item 8

### Ainda em aberto (ordem de prioridade)

1. **Validar empiricamente $B_{NX}(K)$ e $B_{NX,i}(K)$** com os três casos sintéticos (Seção 9.2 e 9.4) — fórmulas corrigidas contra a fonte primária, ainda não testadas em código. Bloqueia toda a F2.
2. Executar a sequência operacional completa do dataset (Seção 5, passos 1–10) para descobrir o $N$ real e congelar `dataset.json` — antes do pipeline de métricas, pode correr em paralelo com engenharia de renderização/infraestrutura.
3. Resolver acesso ao Gemma 2B (gated model) na Fase 0.
4. Confirmar por profiling (Seção 8.4) se o overlay de DOM é de fato o gargalo de FPS, antes de investir tempo na substituição por sprites de canvas.

Depois desses quatro pontos — e, especialmente, depois de rodar a construção real do dataset e validar $B_{NX,i}(K)$ em código — o protocolo experimental pode ser considerado congelado. O próximo passo não é mais revisão textual: é executar a sequência do dataset e a suíte de testes sintéticos de co-ranking.

---

## Referências

- CAI, Xingyu et al. Isotropy in the Contextual Embedding Space: Clusters and Manifolds. **ICLR**, 2021.
- ETHAYARAJH, Kawin. How Contextual are Contextualized Word Representations? **EMNLP-IJCNLP**, 2019.
- LEE, John A.; VERLEYSEN, Michel. Quality assessment of dimensionality reduction: Rank-based criteria. **Neurocomputing**, v. 72, p. 1431–1443, 2009.
- LUEKS, W. et al. How to Evaluate Dimensionality Reduction? Improving the Co-ranking Matrix. **arXiv:1110.3917**, 2011.
- MCINNES, Leland et al. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. **arXiv**, 2018.
- PEARSON, Karl. On lines and planes of closest fit. **Philosophical Magazine**, 1901.
- ROUSSEEUW, Peter J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. **J. Comput. Appl. Math.**, 1987.
- TENENBAUM, Joshua B. et al. A global geometric framework for nonlinear dimensionality reduction. **Science**, 2000.
- VAN DER MAATEN, Laurens; HINTON, Geoffrey. Visualizing data using t-SNE. **JMLR**, 2008.
- VENNA, Jarkko; KASKI, Samuel. Local multidimensional scaling. **Neural Networks**, 2006.
