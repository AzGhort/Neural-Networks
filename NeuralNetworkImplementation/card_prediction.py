import json as js
import numpy as np
import bp as bp

class CardConverter:

    def __init__(self, data_files):
        self.filtered = []
        self.card_inds = {}
        self.game_histories = []
        self.inputs = []
        self.outputs = []
        self.prepare_data(data_files)
        self.set_training_data()
        self.NN = bp.NeuralNetwork([self.cards_count, self.cards_count, self.cards_count])

    # region Neural network
    # train network
    def train_network(self, epochs, pos_iters, learning_rate):
        print("Learning of neural network has started...")
        print("---------------------------------------")
        err = self.get_total_mean_squared_error()
        print("Total mean squared error: {0}".format(err))
        for i in range(0, epochs):
            print("Epoch number {0} has begun.".format(i))
            for (x, y) in zip(self.inputs, self.outputs):
                self.NN.SGD([(x, y)], pos_iters, 1, learning_rate)
                err = self.get_total_mean_squared_error()
            print("Total mean squared error: {0}".format(err))
            print("---------------------------------------")
        print("Learning of neural network completed.")
        print("---------------------------------------")

    def get_vector_mean_squared_error(self, in_vec, out_vec):
        out = self.NN.feedforward(in_vec)
        return np.sum(np.square(np.subtract(out_vec, out)))

    def get_total_mean_squared_error(self):
        counter = 0
        for (i, o) in zip(self.inputs, self.outputs):
            counter = counter + self.get_vector_mean_squared_error(i, o)
        return counter

    def test_cards(self, cards):
        raw_out = converter.NN.feedforward(self.get_vector_from_cards(cards))
        c = self.get_cards_from_vector(raw_out)
        print("Output cards")
        print(c)

    # endregion

    # region Data preparation
    def set_training_data(self):
        for gh in self.game_histories:
            for h in gh:
                self.set_inputs_outputs(h)

    def set_inputs_outputs(self, history):
        for i in range(2, len(history) - 2, 1):
            in_cards = history[i-2:i]
            out_cards = history[i-1:i + 1]
            self.inputs.append(self.get_vector_from_cards(in_cards))
            self.outputs.append(self.get_vector_from_cards(out_cards))

    def get_vector_from_cards(self, cards):
        array = np.zeros((self.cards_count, 1))
        for card in cards:
            array[self.get_number_from_card(card)] = 1
        return array

    def get_cards_from_vector(self, vector):
        cards = []
        ind = np.where(vector == np.amax(vector))
        print(ind[0][0])
        cards.append(self.get_card_from_number(ind[0][0]))
        return cards

    def prepare_data(self, data_files):
        counter = 0
        for filename in data_files:
            (gh, data) = self.get_card_histories(self.get_game_data(filename))
            self.game_histories.append(gh)
            for card in data:
                if (card not in self.filtered):
                    self.filtered.append(card)
                    self.card_inds[card] = counter
                    counter = counter + 1
        self.cards_count = counter

    def get_game_data(self, filename):
        with open(filename) as f:
            data = js.load(f)
        return data

    def get_card_histories(self, data):
        his = [game["card_history"] if (game["hero"] == "Rogue") else [] for game in data["games"]]
        game_histories = []
        all_cards = []
        # ungoro, kobolds and catacombs, classic, classic, knights of frozen throne, witchwood, classic, rastakhan, boomsday
        standard_expansions = ["UNG", "LOO", "EX1", "CS1", "ICC", "GIL", "NEW", "TRL", "BOT", "CS2"]
        for h in his:
            my_cards = []
            op_cards = []
            for card in h:
                id = card["card"]["id"][0:3]
                if (id in standard_expansions):
                    if (card["player"] == "me"):
                        my_cards.append(card["card"]["name"])
                        all_cards.append(card["card"]["name"])
            if (len(h) > 3):
                game_histories.append(my_cards)
                #game_histories.append(op_cards)
        return (game_histories, all_cards)

    def get_number_from_card(self, card):
        return self.card_inds[card]

    def get_card_from_number(self, number):
        return self.filtered[number]
    # endregion

# MAIN
data_files = ["data1.json","data2.json", "data3.json", "data4.json", "data5.json"]
converter = CardConverter(data_files)

print(converter.cards_count)
converter.train_network(5, 3, 1.0)
#converter.test_cards(['Wandering Monster'])
while (True):
    s = input("Enter card name\n")
    converter.test_cards([s])
