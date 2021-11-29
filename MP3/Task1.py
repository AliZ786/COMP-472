import gensim.downloader as api
import csv

def modelEvaluation(model_name):

    # load model
    corpus = api.load(model_name)
    
    with open('MP3/synonyms.csv', newline='') as synonyms_csv:
        reader = csv.reader(synonyms_csv, delimiter=',')
        # skip first row
        next(reader)
        
        details_csv = open(model_name + '-details.csv', 'w')
        guess_word = ''
        label= ''

        for row in reader:
            question_word = row[0]
            correct_word = row[1]
            word_zero = row[2]
            word_one = row[3]
            word_two = row[4]
            word_three = row[5]

            try:
                corpus[question_word]

                # computing the cosine similarity
                similarity_zero = corpus.similarity(question_word, word_zero)
                similarity_one = corpus.similarity(question_word, word_one)
                similarity_two = corpus.similarity(question_word, word_two)
                similarity_three = corpus.similarity(question_word, word_three)

                # finding the word with highest similarity
                highest_similarity = max(similarity_zero, similarity_one, similarity_two, similarity_three)

                # determining the guess word
                if highest_similarity == similarity_zero:
                    guess_word = word_zero
                elif highest_similarity == similarity_one:
                    guess_word = word_one
                elif highest_similarity == similarity_two:
                    guess_word = word_two
                elif highest_similarity == similarity_three:
                    guess_word = word_three

                if guess_word == correct_word:
                    label = 'correct'
                else:
                    label = 'wrong'    

            except:
                print(f'The word {question_word} was not found in this model.')
                label = 'guess'
            
            # Task 1.1
            details_csv.write(f'{question_word},{correct_word},{guess_word},{label}\n')
        
        details_csv.close()


if __name__ == '__main__':
    modelEvaluation(model_name = "word2vec-google-news-300")