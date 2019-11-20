'''
Naive Bayes

Usage:

    python NaiveBayes.py [arguments]

These are the possible arguments:
    --training_data the path to the training data
        default value is 'compas-scores-two-years.csv'
    --features Comma seperated features for the classifier to pay attention to.
        For example, 'age,sex'. Default is 'sex'.
    --ksmooth add k smoothing value, defaults to 0 (no smoothing)

Big Problem: Always predict no recidivism.
    Potential Cause: because chance of recidivism is below 50% for almost every
        feature:value pair, we will almost never predict recidivism. The
        problem gets worse the more features we add, because we treat features
        as independent.
        If for example we split on sex, we see that Males have a 0.446 probability
        and Females a 0.456 probability of recidivating. Note that as both these
        probabilities are below 0.5, we will never predict a person will recidivate
        if our only considered feature is sex. Furthermore, including sex will ONLY
        decrease the probability that a person will recidivate according to this
        model.

Other problems:
    test_labels and test data are of different length
    train_data appears to contain a {}
'''
import argparse
from collections import defaultdict
import math
import matplotlib.pyplot as plt

LABEL_CLASS_NAME = 'two_year_recid'

class NB:

    def __init__(self, training_data, train_labels, include_feats, smooth):
        self.training_data = training_data
        self.train_labels = train_labels
        self.include_feats = include_feats
        self.smooth = smooth
        #Ask layla for help--there should be a better way of doing
        self.recid_probs = None
        self.not_probs = None
        self.num_recid = None
        self.num_not = None


    def train(self):
        '''Get the probabilities that a person with a feature with a certain value
        (e.g. sex is Male) will recidivate for every feature:value group in the included
        features.'''
        num_recid = 0 # the number of people who did recidivate
        num_not = 0 # the number of people who did not recidivate
        for label in self.train_labels:
            if label == '0':
                num_not += 1
            else:
                num_recid += 1

        recid_counts = defaultdict(dict) # a dictionary mapping {feature w: value v: #(w has value v| did_recid)}
        not_counts = defaultdict(dict) # a dictionary mapping {feature w: value v: #(w has value v| di_not_recid)}
        total_counts = defaultdict(dict) # a dictionary mapping {feature w: value v #(w is v}
        for i in range(len(self.training_data)):
            person = self.training_data[i]
            for feature, value in person.items():
                if feature in self.include_feats:
                    # increase count in total_counts
                    if value not in total_counts[feature]:
                        total_counts[feature][value] = 1
                    else:
                        total_counts[feature][value] += 1
                    # increment counts in recid_counts or not_counts
                    if self.train_labels[i] == '1':
                        if value not in recid_counts[feature]:
                            recid_counts[feature][value] = 1
                        else:
                            recid_counts[feature][value] += 1
                    else:
                        if value not in not_counts[feature]:
                            not_counts[feature][value] = 1
                        else:
                            not_counts[feature][value] += 1

        #print("counts:", recid_counts)
        #print("total:", recid_total_counts)
        recid_probs = defaultdict(dict) # a dictionary mapping {feature w: value v: P(w has value v| did_recid)}
        not_probs = defaultdict(dict) # a dictionary mapping {feature w: value v: P(w has value v| did_not_recid)}

        total_possible_attributes = 0
        for feature in total_counts:
            # ONLY WORKS IF ALL FEATURE'S VALUES ARE IN BOTH CLASSES
            total_possible_attributes += len(total_counts[feature])

        for feature, inner_dict in total_counts.items():
            for value in inner_dict:
                if value not in recid_counts[feature]:
                    recid_probs[feature][value] = 1 / (total_possible_attributes)
                else:
                    recid_probs[feature][value] = (recid_counts[feature][value] + self.smooth) / (total_counts[feature][value] + total_possible_attributes*self.smooth)

                if value not in not_counts[feature]:
                    not_probs[feature][value] = 1 / (total_possible_attributes)
                else:
                    not_probs[feature][value] = (not_counts[feature][value] + self.smooth) / (total_counts[feature][value] + total_possible_attributes*self.smooth)

        self.recid_probs = recid_probs
        self.not_probs = not_probs
        #print("recid_probs:", self.recid_probs)
        #print("not_probs:", self.not_probs)
        self.num_recid = num_recid
        self.num_not = num_not

    def get_recid_prob(self, person):
        '''Obtain the probability that a person will recidivate'''
        log_prob_recid = 0 #work first in log space
        log_prob_not = 0
        for feature, value in person.items():
            if feature in self.include_feats:
                if (self.recid_probs[feature][value] != 0):
                    log_prob_recid += math.log(self.recid_probs[feature][value])
                if (self.recid_probs[feature][value] != 0):
                    log_prob_not += math.log(self.not_probs[feature][value])

        log_prob_recid += math.log(self.num_recid / len(self.training_data))
        log_prob_not += math.log(self.num_not / len(self.training_data))
        prob = math.exp(log_prob_recid) # return a probability not in log space
        prob_not = math.exp(log_prob_not)
        return prob, prob_not


    def classify(self, person):
        '''Given a person, this method computes the summed log probability of
        recidivism given input document, according to naive bayes. If the
        probability of recidivating exceeds the probability of not recidivating
        then this function returns 1. Otherwise, it returns 0.
        '''
        if self.recid_probs == None or self.not_probs == None:
            print("Tried to classify with untrained model")
            return -1
        else:
            prob, not_prob = self.get_recid_prob(person)
            # get best accuracy if we add 0.075 to prob
            if prob < not_prob:
                return 0
            else:
                return 1

    def decile_score(self, person):
        '''Given a person, this method computes the summed log probability of
        recidivism given input document, according to naive bayes. If the
        probability of recidivating exceeds 0.5, then this function returns 1.
        Otherwise, it returns 0.
        '''
        if self.recid_probs == None:
            print("Tried to classify with untrained model")
            return -1
        else:
            prob, not_prob = self.get_recid_prob(person)
            score1 = int(10*prob - 0.5) + 1
            score2 = int(10*not_prob - 0.5) + 1
            return score1, score2

    def get_accuracy(self, test_data, test_labels):
        '''Prints the accuracy and the number predicted to recidivate given
        labled testing data.'''
        num_correct = 0
        num_pred_recid = 0
        for i in range(len(test_labels)):
            #print("\nPredicting on person:", test_data[i])
            #print("\tWe predict:", self.classify(test_data[i]), "with prob of", self.get_recid_prob(test_data[i]))
            #print("\tGround truth:", test_labels[i])
            if self.classify(test_data[i]) == 1:
                num_pred_recid += 1
            if self.classify(test_data[i]) == int(test_labels[i]):
                num_correct += 1
        print("Accuracy of:", num_correct/len(test_labels))
        print("Number predicted to recidivate:", num_pred_recid)

    def get_distances(self, test, labels):
        distances = []
        for i in range(len(labels)):
            if labels[i] == '1':
                distances.append(10 - self.decile_score(test[i])[0])
            else:
                distances.append(self.decile_score(test[i])[0] - 1)
        return distances


def load_data(filepath, train_split=0.9):
    with open(filepath) as f:
        i = 0
        labels = []
        all_people = []
        features = []
        for line in f:
            person = {}
            split_line = line.split(",")
            split_line[-1] = split_line[-1][:-1]
            add_to_labels = False
            for j in range(len(split_line)):
                if i == 0:
                    features.append(split_line[j])
                else:
                    if features[j] == LABEL_CLASS_NAME:
                        add_to_labels = True
                        if split_line[j] == '':
                            labels.append(0)
                        else:
                            labels.append(split_line[j])
                    else:
                        person[features[j]] = split_line[j]
            all_people.append(person)
            if not add_to_labels:
                print("at index", i, "did not add to labels")
            i += 1

        split_index = int(len(all_people)*train_split)
        # returns training set,            training lables,        testing set,             testing labels
        return all_people[:split_index], labels[:split_index], all_people[split_index:], labels[split_index:]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data',
                        type=str,
                        default='compas-scores-two-years.csv',
                        help='Path to training data')
    parser.add_argument('--features',
                        type=str,
                        default="sex",
                        help= "Comma seperated features for the classifier to pay attention to.")
    parser.add_argument('--ksmooth',
                        type=int,
                        default=0,
                        help="K-smoothing value, defaults to 0 (no smoothing).")
    return parser.parse_args()



def main():
    args = parse_args()
    train, train_labels, test, test_labels = load_data(args.training_data)
    model = NB(train, train_labels, args.features.split(','), args.ksmooth)
    model.train()
    #print("Why is this happening? len(test), len(test_labels):",len(test), len(test_labels))
    model.get_accuracy(test, test_labels)
    distances = model.get_distances(train+test, train_labels+test_labels)
    bins = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9]
    histogram = plt.hist(distances, bins, density=True)
    plt.show()



if __name__ == "__main__":
    main()
