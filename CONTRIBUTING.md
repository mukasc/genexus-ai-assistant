Contribuindo para o GeneXus AI Assistant

Obrigado pelo interesse em contribuir! Este documento estabelece as diretrizes para garantir que o projeto permaneça organizado, escalável e fácil de manter.

📂 Arquitetura do Projeto

Recentemente, o projeto passou por uma reestruturação. Por favor, respeite a nova organização de pastas:

/backend: Todo o código fonte Python, API (FastAPI) e regras de negócio ficam aqui. Não crie arquivos .py na raiz.

/monitoring: Configurações da stack PLG (Promtail, Loki, Grafana).

/data: Armazenamento persistente (Banco vetorial, Logs locais, Uploads).

/_legacy: Scripts antigos e descontinuados.

🚀 Como Rodar o Ambiente

Para garantir consistência, utilize os scripts de automação:

Configuração Inicial:
Execute o script start.sh na raiz. Ele criará o ambiente virtual e instalará as dependências corretas.

./start.sh


Variáveis de Ambiente:
Nunca suba chaves de API para o Git. Utilize o .env dentro da pasta backend/. Use o backend/.env.example como base.

💻 Padrões de Código

Python (Backend)

Seguimos a PEP 8.

Utilize Type Hints (tipagem) nas funções sempre que possível.

O servidor é assíncrono (async/await), evite operações bloqueantes na thread principal.

📝 Logs e Observabilidade (CRÍTICO)

Como utilizamos Grafana e Loki para monitoramento, o formato dos logs é crucial:

❌ NUNCA use print(). O print não gera metadados e perde-se no Loki.

✅ USE o logger configurado.

Exemplo correto:

logger.info("Processando arquivo PDF", extra={"filename": "doc.pdf", "user_id": "123"})


Sempre passe dados estruturados no parâmetro extra para facilitar a criação de gráficos no Grafana.

🐳 Docker e Infraestrutura

Se você alterar configurações do Docker (docker-compose.yml), certifique-se de:

Não expor portas desnecessárias.

Usar volumes nomeados ou mapeamentos locais conforme definido em infra_data/.

Testar se o container sobe sem erros com docker-compose up --build.

🔄 Fluxo de Trabalho (Git)

Crie uma branch para sua feature ou correção: git checkout -b feature/nova-funcionalidade.

Faça commits semânticos (ex: feat: adiciona suporte a logs JSON, fix: corrige erro de ingestão).

Abra um Pull Request descrevendo o que foi alterado.

⚠️ White Label

Ao alterar configurações visuais ou de identidade, não altere o código fonte diretamente ("hardcoded"). Use o arquivo backend/app_config.json ou as variáveis de ambiente para garantir que o sistema continue White Label.

Dúvidas? Entre em contato com a equipe de engenharia.