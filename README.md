# Boon AI Hackathon: Document Processing & Entity Mapping

A solution for extracting structured data from documents and mapping entities to a database, handling renamed/acquired entities.

## Quick Setup

1. Clone the repository:
```bash
git clone https://github.com/bennysun1/boon-hackathon-entity-mapping.git
cd boon-hackathon-entity-mapping
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
python src/main.py --input path/to/document.pdf --output results
```

## Project Structure

```
boon-hackathon-entity-mapping/
├── data/                  # Sample data and database files
├── src/                   # Source code
│   ├── document_processor/ # Document processing modules
│   ├── entity_mapper/     # Entity mapping modules
│   ├── models/            # ML model connectors
│   ├── utils/             # Utility functions
│   ├── config.py          # Configuration
│   └── main.py            # Main application entry point
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Test files
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore file
└── requirements.txt       # Project dependencies
```
