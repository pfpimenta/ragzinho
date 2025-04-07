# ragzinho

Mini projeto pra aprendermos a usar RAGs, rodando localmente

## How to run

Install Python dependencies:
´´´bash
pip3 install -r requirements.txt
´´´

If you don't have Ollama installed, install it like this:
´´´bash
sudo curl -fsSL https://ollama.com/install.sh | sh
´´´

Use Ollama to pull the LLM used to your computer:
´´´bash
ollama pull llama2
´´´

Then, run the script:
´´´bash
python3 rag.py
´´´


## TODOs
* rever se precisamos de todas as dependencias no requirements.txt (provavelmente nao precisamos das com openai no nome)
* fazer rodar na GPU com CUDA
* fazer funcionar com varios documentos
* entender/melhorar o metodo de chunking/parsing de cada documento