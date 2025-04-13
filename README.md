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
# On Linux/Mac:
bash setup.sh

# On Windows:
setup.bat
```

3. Configure your API keys in the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Prepare your example documents by copying them to the appropriate directory:
```bash
# Create a directory for your image files
mkdir -p data/examples

# Copy your example documents to the examples directory
cp /path/to/your/documents/*.jpg data/examples/
```

## Running the Solution

Process documents and map entities:

```bash
python src/main.py process data/examples --output results
```

This will:
1. Extract entities from your documents using vision-enabled LLMs
2. Map the extracted entities to the mock database with fuzzy matching
3. Detect entity name changes (e.g., through acquisitions or rebranding)
4. Generate detailed reports and visualizations

View the results in the `results/` directory. Each document will have its own subdirectory with:
- `extracted_entities.json`: Structured data extracted from the document
- `mapping_results.json`: Entity mapping results with confidence scores
- HTML visualization of the results

## Options

```
Usage: python src/main.py process [OPTIONS] INPUT_PATH

  Process documents and map entities to the database.

Options:
  --output-dir TEXT            Directory to save results
  --model TEXT                 LLM model to use for extraction
  --detail-level TEXT          Detail level for vision models (high, medium, low)
  --db-path TEXT               Path to entity database JSON file
  --match-threshold FLOAT      Threshold for entity matching confidence
  -v, --verbose                Enable verbose output
  --help                       Show this message and exit
```

## Example Documents

The project includes examples from the logistics/transportation domain:

1. **Freight Invoice from Steve's Trucking** - This contains information about a shipping transaction with customer details, rates, and payment information.

2. **Rate Confirmation Sheet from Bennett International Logistics** - This shows carrier details, load information, origin/destination, and special instructions.

3. **Bill of Lading from Linbis Logistics Software** - This is a shipping document that includes shipper, carrier, and consignee information.

These documents demonstrate the entity mapping challenge, including:
- Company name variations (e.g., "Steve's Trucking" vs "Steve Trucking Company")
- Corporate affiliations (e.g., "Bennett International Logistics, LLC" being a division of "BENNETT TRUCK TRANSPORT, LLC")
- Abbreviated company names (e.g., "GT XPRESS INC" vs "GT Express Incorporated")

## Project Structure

```
boon-hackathon-entity-mapping/
├── data/                  # Sample data and database files
│   ├── db/                # Mock entity database
│   └── examples/          # Example document images
├── src/                   # Source code
│   ├── document_processor/ # Document processing modules
│   ├── entity_mapper/     # Entity mapping modules
│   ├── models/            # ML model connectors
│   ├── utils/             # Utility functions
│   ├── config.py          # Configuration
│   └── main.py            # Main application entry point
├── results/               # Output directory for processed results
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore file
├── README.md              # Project documentation
├── requirements.txt       # Project dependencies
├── setup.bat              # Windows setup script
└── setup.sh               # Unix/Linux/MacOS setup script
```
