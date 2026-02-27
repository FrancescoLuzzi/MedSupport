# ⚙️ Variables
PORT := "4433"
CERT := "cert.pem"
KEY := "key.pem"

# 📋 Default: list available commands
default:
    @just --list

# 📥 Download CSV files from AIFA source list
[group('model')]
download:
    @python scripts/download_list.py

# 🧠 Train and export ONNX models
[group('model')]
train:
    @python scripts/train.py
    @echo "🔗 Creating ONNX symlinks in final folder..."

# 🔍 Run similarity comparison examples
[group('model')]
compare:
    @python scripts/compare.py

# 🗂️ Create triplets dataset parquet from CSV
[group('model')]
dataset:
    @python scripts/create_dataset.py

# 🏗️ Build Turso-compatible vector DB from triplets.parquet
[group('model')]
build-vector-db:
    @python scripts/build_vector_db.py

# 🔐 Generate self-signed SSL certificates
[group('frontend')]
cert:
    @echo "⏳ Creating development certificates..."
    @openssl req -x509 -newkey rsa:4096 -keyout {{KEY}} -out {{CERT}} -sha256 -days 365 -nodes -subj "/C=XX/ST=State/L=City/O=Development/OU=IT/CN=localhost" 2> /dev/null
    @echo "✅ Certificates generated: {{CERT}} and {{KEY}}"

# 🚀 Serve frontend and open test page
[group('frontend')]
serve-test:
    @if [ ! -f {{CERT}} ]; then \
        echo "❌ Missing certs, generating..."; \
        just cert; \
    fi
    @echo "🚀 Starting Vite dev server on https://localhost:{{PORT}}"
    @open https://localhost:{{PORT}}/frontend/index.html
    PORT={{PORT}} CERT={{CERT}} KEY={{KEY}} bun run dev

# 🧹 Clean up certificate files
[group('frontend')]
clean:
    rm -f {{CERT}} {{KEY}}
    @echo "🗑️  Certificates removed."

[group('frontend')]
lint:
    @cd frontend && npm run lint

[group('frontend')]
format:
    @cd frontend && npm run format
