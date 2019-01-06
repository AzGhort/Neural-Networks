import json as js


class CardConverter:

    def __init__(self, filename):
        self.filtered = []
        self.card_inds = {}
        counter = 0
        (names, data) = self.get_card_histories(self.get_data(filename))
        for card in data:
            if (card not in self.filtered):
                self.filtered.append(card)
                self.card_inds[card] = counter
                counter = counter + 1
        self.cards_count = counter

    def get_data(self, filename):
        with open(filename) as f:
            data = js.load(f)
        return data

    def get_card_histories(self, data):
        his = [game["card_history"] for game in data["games"]]
        game_histories = []
        all_cards = []
        # ungoro, kobolds and catacombs, classic, classic, knights of frozen throne, witchwood, classic, rastakhan, boomsday
        standard_expansions = ["UNG", "LOO", "EX1", "CS1", "ICC", "GIL", "NEW", "TRL", "BOT"]
        for h in his:
            my_cards = []
            op_cards = []
            for card in h:
                id = card["card"]["id"][0:3]
                if (id in standard_expansions):
                    if (card["player"] == "opponent"):
                        op_cards.append(card["card"]["name"])
                    else:
                        my_cards.append(card["card"]["name"])
                    all_cards.append(card["card"]["name"])
            game_histories.append((my_cards, op_cards))
        return (game_histories, all_cards)

    def get_number_from_card(self, card):
        return self.card_inds[card]

    def get_card_from_number(self, number):
        return self.filtered[number]




converter = CardConverter("data.json")
for i in range(0, converter.cards_count):
    print(converter.get_card_from_number(i))
