🤖 GeneXus AI Assistant (Enterprise Edition)

O GeneXus AI Assistant é uma plataforma de Inteligência Artificial generativa White Label focada em RAG (Retrieval-Augmented Generation). Ele permite criar assistentes que conversam com documentos PDF e websites, com uma arquitetura robusta preparada para escala e observabilidade total.

🚀 Principais Funcionalidades

🧠 RAG Avançado: Ingestão de PDFs e Scraping de Sites para compor a base de conhecimento.

⚡ Backend de Alta Performance: API construída em FastAPI (Assíncrono) para baixa latência.

🎨 White Label Nativo: Personalização de identidade (nome, cores, prompts) via configuração JSON, sem alterar código.

📊 Observabilidade Enterprise (Stack PLG):

Promtail & Loki: Coleta e agregação de logs estruturados.

Grafana: Dashboards em tempo real de latência, erros e consumo de tokens.

Logs JSON: Rastreabilidade completa com metadados para auditoria.

💎 Vetores Otimizados: Uso do ChromaDB com cache de embeddings para reduzir custos e latência.

🛡️ Segurança: Gerenciamento de segredos via .env e validação de requisições.

📂 Estrutura do Projeto

O projeto segue uma arquitetura limpa de microsserviços/módulos:

/
├── backend/                  # Cérebro da aplicação (API FastAPI)
│   ├── server.py             # Entrypoint da API
│   ├── app_config.json       # Configurações visuais (White Label)
│   ├── requirements.txt      # Dependências Python
│   └── .env                  # Segredos (API Keys)
│
├── monitoring/               # Infraestrutura de Observabilidade
│   ├── loki-config.yaml      # Configuração do Agregador de Logs
│   ├── promtail-config.yaml  # Configuração do Coletor de Logs
│   └── grafana-datasources   # Conexão automática Grafana -> Loki
│
├── data/                     # Persistência de Dados (Ignorado no Git)
│   ├── chroma_db/            # Banco Vetorial
│   └── uploads/              # Arquivos temporários
│
├── infra_data/               # Volumes Docker persistentes (Logs/Dashboards)
├── _legacy/                  # Scripts antigos (Quarentena)
├── docker-compose.yml        # Orquestração de Containers
├── start.sh                  # Script de Inicialização Automatizada
└── README.md                 # Você está aqui


🛠️ Pré-requisitos

Docker e Docker Compose (Para rodar a stack completa).

Python 3.10+ (Para rodar apenas o backend localmente).

Uma chave de API do Google Gemini (Obtenha no Google AI Studio).

🏁 Como Iniciar (Quick Start)

Utilize o script automatizado start.sh para configurar o ambiente em segundos.

1. Configuração Inicial

# Dê permissão de execução ao script
chmod +x start.sh

# Execute o assistente de instalação
./start.sh


O script irá:

Verificar/Criar o arquivo .env em backend/.

Criar um ambiente virtual Python (venv).

Instalar as dependências.

Oferecer um menu de execução.

2. Editando as Configurações

Antes de rodar, edite o arquivo backend/.env e adicione sua chave:

GEMINI_API_KEY=sua_chave_aqui_xyz
CHROMA_DB_PATH=../data/chroma_db


3. Executando a Aplicação

Opção A: Infraestrutura Completa (Docker) 🐳

Recomendado para simular produção com logs e monitoramento.
No menu do start.sh, escolha a opção 2.

API (Backend): http://localhost:8001

Grafana (Dashboards): http://localhost:3001 (Login: admin / admin)

Documentação (Swagger): http://localhost:8001/docs

Opção B: Desenvolvimento Local (Python Puro) 🐍

Recomendado para depuração rápida de código.
No menu do start.sh, escolha a opção 1.

API (Backend): http://localhost:8001

📡 Endpoints da API

A API documentada via Swagger/OpenAPI está disponível em /docs. Os principais endpoints são:

Método

Endpoint

Descrição

POST

/api/chat

Envia uma mensagem e recebe resposta com contexto (RAG).

POST

/api/ingest-pdf

Faz upload e processamento vetorial de PDFs.

POST

/api/ingest-url

Faz scraping e processamento de páginas web.

GET

/api/config

Retorna configurações de UI (Cores, Logo) do White Label.

GET

/api/health

Verifica status do Banco Vetorial e API Key.

📊 Observabilidade (Logs e Métricas)

O sistema utiliza a stack PLG. Para acessar os logs:

Acesse o Grafana em http://localhost:3001.

Vá em Explore.

Selecione a fonte de dados Loki.

Use queries LogQL para filtrar, ex: {app="genexus-ai"}.

Dica: Os logs são estruturados em JSON. Evite usar print() no código; use sempre logger.info() passando o argumento extra={...} para que os dados apareçam como campos filtráveis no Grafana.

🎨 Personalização White Label

Para alterar a identidade do assistente sem mexer no código, edite o arquivo backend/app_config.json:

{
  "identity": {
    "app_name": "Meu Assistente Corporativo",
    "primary_color": "#FF5722",
    "welcome_message": "Olá! Como posso ajudar sua empresa hoje?"
  },
  "llm": {
    "model_name": "gemini-2.0-flash-exp",
    "temperature": 0.2
  }
}


🤝 Como Contribuir

Consulte o arquivo CONTRIBUTING.md para diretrizes sobre estilo de código, fluxo de commits e padrões de arquitetura.

📄 Licença

Este projeto é licenciado sob a GNU Affero General Public License v3.0 (AGPLv3) - consulte o arquivo LICENSE para detalhes.
Isso garante que modificações disponibilizadas via rede (SaaS) devem ter seu código-fonte compartilhado.