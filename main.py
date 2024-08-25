import random
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
from tqdm import tqdm
import optuna
from optuna import Trial

# Definicje wartości kart i początkowy system liczenia Uston SS
CARD_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10,
               'A': 11}
START_COUNT = -26


class Action(Enum):
    STAND = "STAND"
    HIT = "HIT"
    DOUBLE = "DOUBLE"
    SPLIT = "SPLIT"
    SURRENDER = "SURRENDER"


class Shoe:
    def __init__(self, num_decks):
        self.num_decks = num_decks
        self.deck = self._create_shoe()
        self.used_cards = 0

    def _create_shoe(self):
        deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4 * self.num_decks
        random.shuffle(deck)
        return deque(deck)

    def draw_card(self):
        if not self.deck:
            self.deck = self._create_shoe()
            self.used_cards = 0
        card = self.deck.popleft()
        self.used_cards += 1
        return card

    def reset(self):
        self.deck = self._create_shoe()
        self.used_cards = 0

    def cards_remaining_ratio(self):
        total_cards = 52 * self.num_decks
        return (total_cards - self.used_cards) / total_cards


class Hand:
    def __init__(self):
        self.cards = []
        self.is_split_aces = False
        self.is_split = False

    def add_card(self, card):
        self.cards.append(card)

    def value(self):
        value = sum([CARD_VALUES[card] for card in self.cards])
        if value > 21 and 'A' in self.cards:
            value -= 10
        return value

    def is_blackjack(self):
        return len(self.cards) == 2 and self.value() == 21 and not self.is_split_aces

    def can_split(self):
        return len(self.cards) == 2 and self.cards[0] == self.cards[1] and not self.is_split_aces

    def can_double(self):
        return len(self.cards) == 2 and not self.is_split_aces

    def soft_hand(self):
        return 'A' in self.cards and sum([CARD_VALUES[card] for card in self.cards]) - 10 * self.cards.count('A') <= 10

    def can_surrender(self):
        return len(self.cards) == 2 and not self.is_split


class Player:
    def __init__(self, shoe, uston_ss, start_count):
        self.shoe = shoe
        self.hands = [Hand()]
        self.count = start_count
        self.uston_ss = uston_ss

    def _update_count(self, card):
        self.count += self.uston_ss[card]

    def draw_card(self):
        card = self.shoe.draw_card()
        self._update_count(card)
        return card

    def play_hand(self, game, hand_index, dealer_upcard, bet):
        hand = self.hands[hand_index]
        bankroll_change = 0
        insurance_bet = 0

        if dealer_upcard == 'A':
            insurance_bet = bet / 2

        while True:
            if hand.is_split_aces:
                break

            action = game.basic_strategy(hand, dealer_upcard)
            if action == Action.STAND:
                break
            elif action == Action.HIT:
                hand.add_card(self.draw_card())
                if hand.value() > 21:
                    break
            elif action == Action.DOUBLE:
                bet *= 2
                hand.add_card(self.draw_card())
                break
            elif action == Action.SPLIT:
                self.hands.append(Hand())
                split_card = hand.cards.pop()
                self.hands[-1].add_card(split_card)
                hand.add_card(self.draw_card())
                self.hands[-1].add_card(self.draw_card())
                self.hands[-1].is_split = True
                hand.is_split = True
                if split_card == 'A':
                    hand.is_split_aces = True
                    self.hands[-1].is_split_aces = True
                bankroll_change += self.play_hand(game, len(self.hands) - 1, dealer_upcard, bet)
            elif action == Action.SURRENDER:
                return bankroll_change - bet / 2

        dealer_hand = Hand()
        dealer_hand.add_card(dealer_upcard)
        dealer_hand.add_card(self.draw_card())
        while dealer_hand.value() < 17:
            dealer_hand.add_card(self.draw_card())

        if dealer_hand.is_blackjack():
            bankroll_change += 2 * insurance_bet
        else:
            bankroll_change -= insurance_bet

        # Sprawdzenie czy gracz lub dealer mają blackjacka
        if hand.is_blackjack() and not dealer_hand.is_blackjack():
            return bankroll_change + 1.5 * bet
        elif dealer_hand.is_blackjack() and not hand.is_blackjack():
            return bankroll_change - bet
        elif hand.is_blackjack() and dealer_hand.is_blackjack():
            return bankroll_change

        if hand.value() > 21:
            return bankroll_change - bet
        elif dealer_hand.value() > 21 or hand.value() > dealer_hand.value():
            return bankroll_change + bet
        elif hand.value() < dealer_hand.value():
            return bankroll_change - bet
        else:
            return bankroll_change


class Dealer:
    def __init__(self):
        self.hand = Hand()


class BlackjackGame:
    def __init__(self, num_decks, bet_mapping, shoe_ratio):
        self.num_decks = num_decks
        self.bet_mapping = bet_mapping
        self.shoe_ratio = shoe_ratio
        self.count_results = defaultdict(lambda: {'total': 0, 'count': 0})

    def basic_strategy(self, hand, dealer_upcard):
        player_value = hand.value()
        dealer_value = CARD_VALUES[dealer_upcard]

        if hand.can_surrender():
            if player_value == 16 and dealer_value in [9, 10]:
                return Action.SURRENDER
            if player_value == 15 and dealer_value == 10:
                return Action.SURRENDER

        if hand.can_split():
            if hand.cards[0] in ['A', '8']:
                return Action.SPLIT
            elif hand.cards[0] in ['2', '3', '7'] and dealer_value in range(2, 8):
                return Action.SPLIT
            elif hand.cards[0] == '6' and dealer_value in range(2, 7):
                return Action.SPLIT
            elif hand.cards[0] == '4' and dealer_value in range(5, 7):
                return Action.SPLIT
            elif hand.cards[0] == '9' and dealer_value in [2, 3, 4, 5, 6, 8, 9]:
                return Action.SPLIT

        if hand.soft_hand():
            if hand.can_double():
                if dealer_upcard == 6 and player_value in range(13, 19):
                    return Action.DOUBLE
                if dealer_upcard == 5 and player_value in range(13, 18):
                    return Action.DOUBLE
                if dealer_upcard == 4 and player_value in range(15, 18):
                    return Action.DOUBLE
                if dealer_upcard == 3 and player_value in range(17, 19):
                    return Action.DOUBLE
                if dealer_upcard == 2 and player_value == 18:
                    return Action.DOUBLE

            if player_value >= 19:
                return Action.STAND
            if player_value == 18:
                return Action.STAND if dealer_value < 9 else Action.HIT

            return Action.HIT

        if player_value >= 17:
            return Action.STAND
        elif player_value >= 13:
            return Action.STAND if dealer_value < 7 else Action.HIT
        elif player_value == 12:
            return Action.STAND if dealer_value in [3, 4, 5, 6] else Action.HIT
        elif player_value == 11:
            return Action.DOUBLE if hand.can_double() else Action.HIT
        elif player_value == 10:
            return Action.DOUBLE if dealer_value < 10 and hand.can_double() else Action.HIT
        elif player_value == 9:
            return Action.DOUBLE if dealer_value in [3, 4, 5, 6] and hand.can_double() else Action.HIT
        else:
            return Action.HIT

    def play_round(self, player):
        if player.shoe.cards_remaining_ratio() < self.shoe_ratio:
            player.shoe.reset()
            player.count = START_COUNT

        player.hands = [Hand()]
        dealer = Dealer()

        player.hands[0].add_card(player.draw_card())
        player.hands[0].add_card(player.draw_card())
        dealer.hand.add_card(player.draw_card())

        initial_bet = 0
        for threshold in sorted(self.bet_mapping.keys()):
            if player.count >= threshold:
                initial_bet = self.bet_mapping[threshold]

        count = player.count
        round_result = player.play_hand(self, 0, dealer.hand.cards[0], initial_bet)

        self.count_results[count]['total'] += round_result / initial_bet if initial_bet > 0 else 0
        self.count_results[count]['count'] += 1

        return round_result


class BlackjackSimulation:
    def __init__(self, num_players, num_trials, num_decks, shoe_ratio, bet_mapping, uston_ss, start_count):
        self.num_players = num_players
        self.num_trials = num_trials
        self.shoe_ratio = shoe_ratio
        self.uston_ss = uston_ss
        self.start_count = start_count
        self.game = BlackjackGame(num_decks, bet_mapping, shoe_ratio)
        self.results = defaultdict(list)
        self.players = [Player(Shoe(num_decks), self.uston_ss, self.start_count) for _ in range(self.num_players)]

    def run_simulation(self):
        total_iterations = len(self.players) * self.num_trials
        with tqdm(total=total_iterations) as pbar:
            for player_idx, player in enumerate(self.players):
                for _ in range(self.num_trials):
                    bankroll_change = self.game.play_round(player)
                    if self.results[player_idx]:
                        bankroll_change += self.results[player_idx][-1]
                    self.results[player_idx].append(bankroll_change)
                    pbar.update(1)

    def calculate_total_sum(self):
        count_values = sorted(self.game.count_results.keys())
        total_sum = sum(self.game.count_results[count]['total'] for count in count_values)
        return total_sum


# Funkcja optymalizacyjna dla Optuna
def objective(trial: Trial):
    uston_ss = {
        '2': trial.suggest_int('2', -5, 5),
        '3': trial.suggest_int('3', -5, 5),
        '4': trial.suggest_int('4', -5, 5),
        '5': trial.suggest_int('5', -5, 5),
        '6': trial.suggest_int('6', -5, 5),
        '7': trial.suggest_int('7', -5, 5),
        '8': trial.suggest_int('8', -5, 5),
        '9': trial.suggest_int('9', -5, 5),
        '10': trial.suggest_int('10', -5, 5),
        'J': trial.suggest_int('J', -5, 5),
        'Q': trial.suggest_int('Q', -5, 5),
        'K': trial.suggest_int('K', -5, 5),
        'A': trial.suggest_int('A', -5, 5),
    }

    start_count = trial.suggest_int('start_count', -50, 0)

    simulation = BlackjackSimulation(num_players=5, num_trials=200000, num_decks=6,
                                     shoe_ratio=5 / 6, bet_mapping={1: 100, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600},
                                     uston_ss=uston_ss, start_count=start_count)
    simulation.run_simulation()

    return simulation.calculate_total_sum()


# Optymalizacja za pomocą Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1200)

# Wyniki optymalizacji
print("Optymalizowane wartości USTON_SS:", study.best_params)
print("Najlepszy wynik:", study.best_value)
