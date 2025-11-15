import ollama

if __name__ == '__main__':

  client = ollama.Client()
  stream = client.generate(model='llama3.2:3b', prompt='Tell me a joke.', stream=True)

  for chunk in stream:
    print(chunk['response'], end='', flush=True)

