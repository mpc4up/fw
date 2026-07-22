# Tiny House RESEX — pipeline 3D + world model (TD-MPC2)

Notebook único (`gerar_modelo_3d.py`, formato células `# %%` — abre direto no
Colab ou no VS Code) com dois estágios independentes que compartilham a mesma
geometria extraída de `Planta_baixa.svg`:

1. **SVG → 3D (GLB)** — Estágios 0-5: lê `Planta_baixa.svg` e exporta
   `tiny_house.glb` para uso em Three.js.
2. **World model (TD-MPC2)** — Estágios 6-7: constrói um ambiente Gyminasium
   (`TinyHouseWorldEnv`) desta tiny house específica e prepara o ajuste fino
   via [TD-MPC2](https://github.com/nicklashansen/tdmpc2).

Este README cobre o **estágio 2** (world model + ajuste fino) rodando no
Google Colab. Para o pipeline 3D, ver os comentários do próprio
`gerar_modelo_3d.py` (Estágios 0-5).

---

## 1. Abrir no Colab

1. Suba `Planta_baixa.svg`, `Base.md`, `Hidraulica.md`,
   `Paisagismo_Mangueiras.md` e `gerar_modelo_3d.py` para o mesmo diretório
   do runtime do Colab (ou monte o Google Drive e `cd` até a pasta).
2. Importe `gerar_modelo_3d.py` como notebook: **Arquivo → Abrir notebook →
   Upload**, ou converta antes com `jupytext --to notebook
   gerar_modelo_3d.py` se preferir um `.ipynb` de verdade.
3. Ative uma GPU (T4 é suficiente) em **Ambiente de execução → Alterar tipo
   de ambiente de execução** — necessária para o treino do TD-MPC2 no
   Estágio 7, não para os Estágios 0-6 (que rodam em CPU).
4. Rode as células em ordem. O Estágio 0 já instala `shapely`, `trimesh`,
   `mapbox_earcut`, `triangle` e `gymnasium` (`torch` já vem pré-instalado
   no Colab).

## 2. O que o Estágio 6 constrói

`TinyHouseWorldEnv` é um `gym.Env` (API Gymnasium: `reset()` / `step()`)
específico desta casa, com três subsistemas:

| Subsistema | Fonte dos dados | Grau de aterramento |
|---|---|---|
| Hidráulica/saneamento | `Hidraulica.md` (5 redes por cor, 2 reservatórios reais: BET, cisterna) | alto — números reais do memorial |
| Rodízio de irrigação | `Hidraulica.md` §3.5 + `Paisagismo_Mangueiras.md` (4 linhas, 14 rasgos, espécie por rasgo) | alto — resolve um controle que o memorial deixa em aberto ("frequência do rodízio... [DEFINIR NA PRÁTICA]") |
| Segurança/reconhecimento de fauna | nenhuma — extensão pedida na conversa, não documentada nos `.md` | hipótese de design, isolada em `ZONE_AFFINITY` para recalibrar depois |

As 6 zonas internas do ambiente são literalmente `room_polys.keys()` — as
mesmas salas/cápsulas extraídas do SVG nos Estágios 1-2. Nenhuma geometria é
duplicada entre o pipeline 3D e o world model.

**Observação** (118 valores): níveis dos 2 reservatórios + carga das 3 redes
com válvula + umidade/descanso dos 14 rasgos + detecção/confiança/espécie
das 12 zonas + fase do ciclo dia-noite.

**Ação** (20 valores, contínua em `[-1, 1]`): abertura de 3 válvulas +
rasgo-alvo das 4 linhas de irrigação + atenção de 12 câmeras/sensores +
sensibilidade global de alerta.

O Estágio 6d roda sozinho um teste de sanidade (1000 passos com ação
aleatória + `gymnasium.utils.env_checker.check_env`) — se essa célula falhar
no seu ambiente, pare antes de ir para o TD-MPC2.

## 3. Ajuste fino via TD-MPC2 (Estágio 7)

O Estágio 7 **não reimplementa** o TD-MPC2 — o repositório é código de
pesquisa (API muda entre commits), então reimplementar o algoritmo aqui
deixaria de ser "TD-MPC2" de verdade. Em vez disso, ele gera os três
artefatos que o repositório espera:

- `GymToTDMPC2Wrapper` — adapta `TinyHouseWorldEnv` para a interface que os
  scripts de treino do TD-MPC2 usam (`reset()` → obs; `step()` → `(obs,
  reward, done, info)`; atributos `obs_shape`/`action_dim`/`episode_length`).
- `make_tinyhouse_env(...)` — factory no formato que
  `tdmpc2/envs/__init__.py::make_env(cfg)` espera.
- `tdmpc2_tinyhouse_config.yaml` — config Hydra da tarefa, gerado com as
  dimensões **reais** do ambiente (não digitadas à mão): `obs_shape:
  {state: [118]}`, `action_dim: 20`, `episode_length: 500`.

### Passo a passo no Colab

```bash
# 1. clonar o repo oficial e instalar dependências
!git clone https://github.com/nicklashansen/tdmpc2.git
!pip install -q -r tdmpc2/requirements.txt
```

```python
# 2. registrar a tarefa — cole no topo de tdmpc2/envs/__init__.py::make_env(cfg),
#    antes do dispatch por prefixo (dmcontrol-*, mw-*, ...)
if cfg.task == "tinyhouse-resex":
    from gerar_modelo_3d import make_tinyhouse_env
    return make_tinyhouse_env(cfg.episode_length, cfg.seed)
```

```bash
# 3. rodar o treino, apontando para o config gerado no Estágio 7b
python tdmpc2/train.py \
    --config-path=. --config-name=tdmpc2_tinyhouse_config \
    task=tinyhouse-resex \
    checkpoint=/caminho/para/tdmpc2-multitask-5M.pt \
    multitask=false \
    steps=1_000_000
```

> A assinatura exata de `make_env`/`TDMPC2`/`Buffer` pode ter mudado desde a
> escrita deste notebook — confira o commit atual do repo clonado antes de
> rodar; pequenas divergências de versão podem exigir ajuste no
> `GymToTDMPC2Wrapper` (Estágio 7a).

### Duas decisões deixadas em aberto (revisar antes de rodar de verdade)

- **Checkpoint pré-treinado vs. treinar do zero**: os checkpoints
  multi-task oficiais do TD-MPC2 foram pré-treinados em locomoção/manipulação
  (DMControl, Meta-World, MyoSuite) — domínio bem diferente de
  hidráulica/irrigação/segurança. `checkpoint=` no comando acima faz
  fine-tuning a partir deles; **omitir essa linha treina do zero**, que é a
  opção mais segura até haver evidência de que a transferência ajuda.
- **Fonte dos dados de treino**: a config assume RL 100% online (rollouts
  gerados pelo próprio `TinyHouseWorldEnv`, sem dataset prévio). Se surgir
  um dataset real de sensores/hidrômetros da casa, ele entraria como
  *offline pretraining* do replay buffer antes do fine-tuning online — não
  implementado neste notebook.

## 4. Recalibrando com dados de campo

Os únicos números do Estágio 6 que **não** vêm de `Hidraulica.md` ou
`Paisagismo_Mangueiras.md` são as afinidades de fauna por zona
(`ZONE_AFFINITY`) e as constantes de dinâmica (vazões de entrada/saída dos
reservatórios, taxa de secagem do solo, etc., espalhadas no corpo de
`TinyHouseWorldEnv.step`). Estão isoladas e comentadas exatamente para
serem substituídas por observação real assim que houver dados de campo —
nenhuma precisa de mudança estrutural no ambiente.

## 5. Saídas geradas por `gerar_modelo_3d.py`

| Arquivo | Estágio | Conteúdo |
|---|---|---|
| `tiny_house.glb` | 5 | casa (paredes segmentadas, aberturas reais) + terreno/paisagismo, pronto para `THREE.GLTFLoader` |
| `tdmpc2_tinyhouse_config.yaml` | 7b | config Hydra da tarefa `tinyhouse-resex`, dimensões reais do ambiente |
