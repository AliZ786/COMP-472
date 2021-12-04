
import gensim.downloader as api
import csv

def modelEvaluation(model_name, filemode):

    # load model
    corpus = api.load(model_name)

    C = 0
    V = len(open('Files/synonyms.csv').readlines())-1
    
    with open('Files/synonyms.csv', newline='') as synonyms_csv:
        reader = csv.reader(synonyms_csv, delimiter=',')
        # skip first row
        next(reader)
        
        details_csv = open(model_name + '-details.csv', 'w')

        for row in reader:
            
            guess_word = ''
            label= ''

            question_word = row[0]
            correct_word = row[1]
            word_zero = row[2]
            word_one = row[3]
            word_two = row[4]
            word_three = row[5]
                
            # check if guess-words are found in the model
            counter = 0
            for j in range(2,6):
                try:
                    corpus[row[j]]
                    counter += 1
                    
                except KeyError:
                    print(f'The word {row[j]} was not found in this model.')

            if counter == 0:
                label = 'guess'
                V -= 1

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

                if label != 'guess':    
                    if guess_word == correct_word:
                        label = 'correct'
                        C += 1
                    else:
                        label = 'wrong'

            except KeyError:
                print(f'The word {question_word} was not found in this model.')
                label = 'guess'
                guess_word = word_one
                V -= 1

            # Task 1.1
            details_csv.write(f'{question_word},{correct_word},{guess_word},{label}\n')
        
        details_csv.close()

        # Task 1.2
        analysis_csv = open('analysis.csv', filemode)
        analysis_csv.write(f'{model_name},{len(corpus)},{C},{V},{C/V}\n')
        analysis_csv.close()

    synonyms_csv.close()
    
    performanceRatio = round((C/V) * 100, 2)
    return performanceRatio 


if __name__ == '__main__':
    modelEvaluation(model_name = "word2vec-google-news-300", filemode = "w")
