import json
import re
import nltk
import math
import timeit
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import wordnet
from collections import Counter
from collections import defaultdict

nltk.data.path.append('/modules/cs918/nltk_data/')
UNK = "<UNK>"

def preprocess(content):
    urls = re.compile(r"\b(https?://)?([a-z_\-0-9]+)(\.[a-z_\-0-9]+)+(/\S*)*\b")
    content = [re.sub(urls, "", x) for x in content]
    #print ("printing content without urls \n", content[417])

    digits = re.compile(r"\b[0-9]+\b")
    content = [re.sub(digits, "", x) for x in content]
    #print ("printing content without full numbers \n", content[417])

    alpha = re.compile(r"[^a-z0-9 ]")
    content = [re.sub(alpha, "", x) for x in content]
    #print ("printing only alphachars and spaces \n", content[417])

    onechar = re.compile(r"\b\w{1}\b")
    content = [re.sub(onechar, "", x) for x in content]
    #print ("printing word with more than 3 chars \n", content[417])

    return content

def get_wordnet_tag(treebank):
    if treebank.startswith("J"):
        return wordnet.ADJ
    elif treebank.startswith("V"):
        return wordnet.VERB
    elif treebank.startswith("N"):
        return wordnet.NOUN
    elif treebank.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tag_lemmatize(wnl, tagger, raw_tokens):
    wordnet_tags = []
    #for word in raw_tokens:
    #    word, treebank = tagger.tag([word])[0]
    #    wordnet_tags.append((word, get_wordnet_tag(treebank)))
    #return [wnl.lemmatize(word, tag) for word, tag in wordnet_tags]
    return [wnl.lemmatize(word) for word in raw_tokens]

def count_pos_neg(positives, negatives, vocab):
    pos_count = 0
    neg_count = 0
    for token, count in vocab.items():
        if token in positives:
            pos_count += count
        if token in negatives:
            neg_count += count
    return pos_count, neg_count

def stupid_backoff(tri_model, bi_model, uni_model, trigram):
    stupid = 0.4
    x, y, z = trigram
    if tri_model[(x, y)][z] > 0:
        return tri_model[(x, y)][z]/bi_model[x][y]
    elif bi_model[y][z] > 0:
        return stupid * (bi_model[y][z]/sum(bi_model[y].values()))
    elif uni_model[z] > 0:
        return stupid*stupid*(uni_model[z]/sum(uni_model.values()))
    else:
        return stupid*stupid*(uni_model[UNK]/sum(uni_model.values()))

def laplace_prob(model, trigram):
    x, y, z = trigram
    bigram = (x,y)
    #print("prob is {} + 1 / {} + {}".format(model[bigram][z], sum(model[bigram].values()), len(model)))
    return (model[bigram][z] + 1.0) / (sum(model[bigram].values()) + len(model))

def gt_counts(vocab):
    counts = {}
    for count in vocab.values():
        counts[count] = counts.get(count, 0) + 1
    return counts

def smooth_gt_counts(gt):
    smooth_counts = {}
    for freq, count in gt.items():
        if gt.get(freq+1):
            #print ("cstar for {} is ({}+1 * {})/{}".format(freq, freq, gt[freq+1], count))
            cstar = (freq+1)*(gt[freq+1]/count)
        else:
            plus_one = gt[1]/sum(gt.values())
            #print ("cstar for {} is ({}+1 * {})/{}".format(freq, freq, plus_one, count))
            cstar = (freq+1)*(plus_one/count)
        smooth_counts[freq] = cstar
    return smooth_counts

def calc_perplexity(tri_model, bi_model, uni_model, test):
    perplexity = 0.0
    num_trigrams = 0
    unseen = 0
    test_trigrams = Counter()
    for row in test:
        test_trigrams.update(row)

    for trigram, count in test_trigrams.items():
        prob = stupid_backoff(tri_model, bi_model, uni_model, trigram)
        if prob != 0:
            perplexity += count*math.log(prob, 2)
        num_trigrams += count
    #print("perplexity: ", perplexity)
    #assert num_trigrams <= 100
    print("unseen words: ", unseen)
    print ("perplexity before root", perplexity)
    return math.pow(2, -(perplexity/num_trigrams))

def form_sentence(tri_model, bigram, length):
    i = 0
    sentence = [bigram[0], bigram[1]]
    while len(sentence) < length:
        all_trigrams = tri_model[(sentence[i], sentence[i+1])]
        most_likely = sorted(all_trigrams.items(), key=lambda x: x[1], reverse=True)[0]
        sentence.append(most_likely[0])
        i += 1
    return sentence

start = timeit.default_timer()
with open("signal-news1.jsonl") as f:
    json_file = f.readlines()

raw_content = []
for line in json_file:
    raw_content.append(json.loads(line)['content'].lower())

#Part A: Text preprocessing

content = preprocess(raw_content)

raw_tokens = [] # keep lists of content seperate
for l in content:
    raw_tokens.append(l.split())

wnl = nltk.WordNetLemmatizer()
tagger = PerceptronTagger()
tokens = []
for l in raw_tokens:
    tokenized_story = tag_lemmatize(wnl, tagger, l)
    tokens.append(tokenized_story)

#Part B: N-Grams
vocab = []
for story in tokens:
    story_vocab = Counter() 
    for t in story:
        story_vocab[t] += 1
    vocab.append(story_vocab)

#Number of tokens N
print ("Number of tokens: ", sum([sum(v.values()) for v in vocab]))
#Vocabulary size
print ("Size of vocab: ", len({word for v in vocab for word in v.keys()}))

#Top 25 trigrams
all_trigrams = Counter()
story_trigrams = []
story_bigrams = []
for i, story in enumerate(tokens):
    tlist = nltk.trigrams(story)
    trigrams = Counter()
    for trigram in tlist:
        trigrams[trigram] += 1
    all_trigrams.update(trigrams)
    story_trigrams.append(trigrams)

    blist = nltk.bigrams(story)
    bigrams = Counter()
    for bigram in blist:
        bigrams[bigram] += 1
    story_bigrams.append(bigrams)

print("Top 25 trigrams :", sorted(list(all_trigrams.items()), key=lambda x: x[1], reverse=True)[:25])

#Positive and negative counts
with open("positive-words.txt", "r") as pos:
    positives = set()
    for word in pos:
        positives.add(re.sub(r"[^a-z0-9 ]", "", word.rstrip("\n")))

with open("negative-words.txt", "r") as neg:
    negatives = set()
    for word in neg:
        negatives.add(re.sub(r"[^a-z0-9 ]", "", word.rstrip("\n")))

pos_stories = 0
neg_stories = 0
pos_count = 0
neg_count = 0
neutral = 0
for story in vocab:
    local_pos, local_neg = count_pos_neg(positives, negatives, story)
    if local_pos > local_neg:
        pos_stories += 1
    if local_pos < local_neg:
        neg_stories += 1
    if local_pos == local_neg:
        neutral += 1
    pos_count += local_pos
    neg_count += local_neg

print("Number of positive words: {}, Number of negative words: {}".format(pos_count, neg_count))
print("Number of positive stories: {}, Number of negative stories: {}".format(pos_stories, neg_stories))

#gt = gt_counts(trigrams)
#smooth_gt = smooth_gt_counts(gt)

tri_model = defaultdict(lambda: defaultdict(lambda: 0))
bi_model = defaultdict(lambda: defaultdict(lambda: 0))
uni_model = defaultdict(lambda: 0)
for i in range(0, 16000):
    for (x, y, z), count in story_trigrams[i].items():
        tri_model[(x, y)][z] += count
    for (x, y), count in story_bigrams[i].items():
        bi_model[x][y] += count
    for t, count in vocab[i].items():
        if count <= 1:
            uni_model[UNK] += count
        else:
            uni_model[t] += count

bigram = ("is","this")
bigram = tag_lemmatize(wnl, tagger, bigram)
print (*form_sentence(tri_model, bigram, 10), sep=" ")

print("perplexity of model is: ", calc_perplexity(tri_model, bi_model, uni_model, story_trigrams[16000:]))

end = timeit.default_timer()
print ("time taken: ", end - start)

if __name__ == "__main__":
    pass
