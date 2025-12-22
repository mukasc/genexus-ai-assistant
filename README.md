🤖 GeneXus AI Assistant (Enterprise Prototype)

Uma plataforma de RAG (Retrieval-Augmented Generation) robusta e White Label, projetada para ingerir conhecimento técnico (PDFs, Sites, YouTube) e assistir desenvolvedores com respostas precisas e contextualizadas.

🚀 Funcionalidades Principais

🧠 Inteligência & RAG

Multi-Source Ingestion: Suporta upload de PDFs, Links de Websites e Vídeos do YouTube (com fallback inteligente de transcrição).

Memória Persistente: O assistente lembra do contexto da conversa (sessões salvas em disco JSON).

Citação de Fontes: As respostas indicam exatamente quais documentos foram consultados.

Query Expansion: (Opcional) Expande perguntas curtas para melhorar a busca no banco vetorial.

Model Fallback: Sistema de rotação automática de modelos (Gemini Flash -> Pro -> 8b) para evitar erros de cota (429).

🏢 Arquitetura & Gestão

White Label Multi-Perfil: Troque entre perfis (ex: "GeneXus" ↔ "Sefaz") instantaneamente, isolando configurações, cores e bases de conhecimento.

Knowledge Manager: Interface visual para listar, pré-visualizar (debug de texto) e excluir documentos indexados.

Streaming Real-time: Respostas geradas token a token (efeito digitação).

Observabilidade: Logs estruturados prontos para a stack PLG (Promtail/Loki/Grafana).

📂 Estrutura do Projeto

/
├── backend/                  # API FastAPI Modular
│   ├── app/
│   │   ├── api/              # Rotas (Chat, Ingest, Admin, Feedback)
│   │   ├── core/             # Lógica RAG, Embeddings, Memória
│   │   └── ...
│   ├── data/                 # Persistência Local (Ignorado no Git)
│   │   ├── chroma_db/        # Banco Vetorial
│   │   ├── sessions/         # Histórico de Conversas
│   │   └── logs/             # Logs do Sistema
│   ├── tests/                # Testes Automatizados (Pytest)
│   ├── main.py               # Entrypoint
│   └── app_config.json       # Configuração dos Perfis White Label
│
├── frontend/                 # Interface React
│   ├── src/
│   │   ├── components/       # KnowledgeManager, LogsViewer, Toast
│   │   └── App.js            # Lógica Principal
│
├── start.sh                  # Script de Inicialização (Menu Interativo)
├── run_dev.sh                # Script de Inicialização Rápida (Dev)
└── run_tests.sh              # Executor de Testes


🏁 Como Iniciar

Pré-requisitos

Python 3.11+

Node.js & NPM

Chave de API do Google Gemini (GEMINI_API_KEY)

1. Configuração Rápida

Utilize o script automatizado na raiz:

chmod +x start.sh
./start.sh


O script irá:

Criar o ambiente virtual Python (venv).

Instalar dependências do Backend.

Oferecer um menu para iniciar o servidor.

2. Iniciando o Frontend

Em um novo terminal:

cd frontend
npm install  # Apenas na primeira vez
npm start


Acesse em: http://localhost:3000

⚙️ Configuração (White Label)

Edite o arquivo backend/app_config.json para criar novos perfis ou ajustar os existentes.

{
  "active_profile": "genexus",
  "profiles": {
    "genexus": {
      "identity": { "app_name": "GeneXus Bot", "primary_color": "#FF5722" },
      "llm": { "model_name": "gemini-1.5-flash", "fallback_order": [...] },
      "retrieval": { "score_threshold": 0.6 }
    },
    "sefaz": { ... }
  }
}


🧪 Testes e Qualidade

Para garantir que o backend está saudável:

./run_tests.sh


Isso executa a suíte pytest validando endpoints, lógica de RAG (mockada) e configurações.

📄 Licença

Este projeto é licenciado sob a AGPLv3.