from torchtext.data import get_tokenizer

input_file = 'Shakespeare.txt'
output_file = 'Shakespeare_preprocessed.txt'


with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()


sentences = []
for i in range(len(lines)):
    lines[i] = lines[i].strip()

for i in range(len(lines)):
    if lines[i].endswith(':'):
        i += 1
        accumulate = str()
        while i < len(lines) and not lines[i].endswith(':'):
            accumulate += (' ' + lines[i])
            i += 1
        sentences.append(accumulate)

with open(output_file, 'w', encoding='utf-8') as f:
    dat_len = len(sentences) // 2
    for i in range(dat_len):
        f.write(sentences[2 * i] + '\n')
        f.write(sentences[2 * i + 1] + '\n')

print(f"output file :{output_file}")
tokenizer = get_tokenizer("basic_english")
all_tokens = dict()

max_len = 0
for sentence in sentences:
    tokens = tokenizer(sentence)
    max_len = max(max_len, len(tokens))
    for token in tokens:
        all_tokens[token] = all_tokens.get(token, 0) + 1
sorted_tokens = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)
print("vocab size", len(sorted_tokens))
print("max_len", max_len)

# output file :Shakespeare_preprocessed.txt
# vocab size 16997
# max_len 360
