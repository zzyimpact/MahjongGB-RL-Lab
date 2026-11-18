import random
from collections import Counter
from .move_generator import move_generator


class Error(Exception):
    def __init__(self, ErrorInfo):
        self.ErrorInfo = ErrorInfo

    def __str__(self):
        return self.ErrorInfo


class ReferenceJudge:
    def __init__(self, config={}):
        if "seed" in config:
            self.seed = config["seed"]
        else:
            self.seed = None

        self.suit_set = ["s", "h", "c", "d"]
        self.card_scale = [
            "A",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
            "J",
            "Q",
            "K",
        ]
        self.major = None
        self.level = None
        self.agent_names = ["player_%d" % i for i in range(4)]

    def reset(self, level="2", banker_pos=0, major="s"):
        self.point_order = [
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
            "J",
            "Q",
            "K",
            "A",
        ]
        self.Major = ["jo", "Jo"]
        self.level = level
        self.first_round = True  # if first_round, banker is the determined during dealing stage (not pre_determined)
        self.banker_pos = banker_pos
        if major == "r":  # randomly generating major
            self.major = random.sample(self.suit_set, 1)[0]
        else:
            self.major = major
        self.total_deck = [i for i in range(108)]
        random.shuffle(self.total_deck)
        self.covered_card = self.total_deck[100:]
        self.card_todeal = self.total_deck[:100]
        self.player_decks = [[] for _ in range(4)]
        self.player_decks[0] = self.card_todeal[:25]
        self.player_decks[1] = self.card_todeal[25:50]
        self.player_decks[2] = self.card_todeal[50:75]
        self.player_decks[3] = self.card_todeal[75:100]
        self._setMajor()
        self.mv_gen = move_generator(self.level, self.major)
        self.score = 0
        self.history = []
        self.played_cards = [[] for _ in range(4)]
        self.reward = None
        self.done = False
        self.round = 0  # 轮次计数器

        self.round += 1
        self.current = self.banker_pos
        return self._get_obs(self.banker_pos)

    def step(self, response):  # response: dict{'player': player_id, 'action': action}
        self.reward = None
        curr_player = response["player"]
        action = response["action"]
        real_action = self._checkLegalMove(action, curr_player)
        real_action = self._name2id_seq(real_action, self.player_decks[curr_player])
        self._play(curr_player, real_action)
        next_player = (curr_player + 1) % 4
        if len(self.history) == 4:  # finishing a round
            winner = self._checkWinner(curr_player)
            next_player = winner
            if len(self.player_decks[0]) == 0:  # Ending the game
                self._reveal(curr_player, winner)
                self.done = True
        self.round += 1

        self.current = next_player
        return self._get_obs(next_player), self.reward, self.done

    def _raise_error(self, player, info):
        raise Error("Player_" + str(player) + ": " + info)

    def _get_obs(self, player):
        obs = {
            "id": player,
            "deck": [self._id2name(p) for p in self.player_decks[player]],
            "history": [[self._id2name(p) for p in move] for move in self.history],
            "major": self.Major,
            "played": [[self._id2name(p) for p in move] for move in self.played_cards],
        }
        return obs

    def _get_action_options(self, player):
        deck = [self._id2name(p) for p in self.player_decks[player]]
        if len(self.history) == 4 or len(self.history) == 0:  # first to play
            return self.mv_gen.gen_all(deck)
        else:
            tgt = [self._id2name(p) for p in self.history[0]]
            poktype = self._checkPokerType(
                self.history[0], (player - len(self.history)) % 4
            )
            if poktype == "single":
                return self.mv_gen.gen_single(deck, tgt)
            elif poktype == "pair":
                return self.mv_gen.gen_pair(deck, tgt)
            elif poktype == "tractor":
                return self.mv_gen.gen_tractor(deck, tgt)
            elif poktype == "suspect":
                return self.mv_gen.gen_throw(deck, tgt)

    def _done(self):
        return self.done

    def _id2name(self, card_id):  # card_id: int[0, 107]
        NumInDeck = card_id % 54
        if NumInDeck == 52:
            return "jo"
        if NumInDeck == 53:
            return "Jo"
        pokernumber = self.card_scale[NumInDeck // 4]
        pokersuit = self.suit_set[NumInDeck % 4]
        return pokersuit + pokernumber

    def _name2id(self, card_name, deck):
        NumInDeck = -1
        if card_name[0] == "j":
            NumInDeck = 52
        elif card_name[0] == "J":
            NumInDeck = 53
        else:
            NumInDeck = self.card_scale.index(card_name[1]) * 4 + self.suit_set.index(
                card_name[0]
            )
        if NumInDeck in deck:
            return NumInDeck
        else:
            return NumInDeck + 54

    def _name2id_seq(self, card_names, deck):
        id_seq = []
        deck_copy = deck + []
        for card_name in card_names:
            card_id = self._name2id(card_name, deck_copy)
            id_seq.append(card_id)
            deck_copy.remove(card_id)
        return id_seq

    def _play(self, player, cards):
        for card in cards:
            self.player_decks[player].remove(card)
            self.played_cards[player].append(card)
        if len(self.history) == 4:  # beginning of a new round
            self.history = []
        self.history.append(cards)

    def _reveal(self, currplayer, winner):  # 扣底
        if self._checkPokerType(self.history[0], (currplayer - 3) % 4) != "suspect":
            mult = len(self.history[0]) * 2
        else:
            divided, _ = self._checkThrow(
                self.history[0], (currplayer - 3) % 4, check=False
            )
            divided.sort(key=lambda x: len(x), reverse=True)
            if len(divided[0]) >= 4:
                mult = len(divided[0]) * 2
            elif len(divided[0]) == 2:
                mult = 4
            else:
                mult = 2

        publicscore = 0
        for pok in self.covered_card:
            p = self._id2name(pok)
            if p[1] == "5":
                publicscore += 5
            elif p[1] == "0" or p[1] == "K":
                publicscore += 10

        self._reward(winner, publicscore * mult)

    def _setMajor(self):
        if self.major != "n":  # 非无主
            self.Major = (
                [
                    self.major + point
                    for point in self.point_order
                    if point != self.level
                ]
                + [suit + self.level for suit in self.suit_set if suit != self.major]
                + [self.major + self.level]
                + self.Major
            )
        else:  # 无主
            self.Major = [suit + self.level for suit in self.suit_set] + self.Major
        self.point_order.remove(self.level)

    def _checkPokerType(self, poker, currplayer):  # poker: list[int]
        level = self.level
        poker = [self._id2name(p) for p in poker]
        if len(poker) == 1:
            return "single"  # 一张牌必定为单牌
        if len(poker) == 2:
            if poker[0] == poker[1]:
                return "pair"  # 同点数同花色才是对子
            else:
                return "suspect"  # 怀疑是甩牌
        if len(poker) % 2 == 0:  # 其他情况下只有偶数张牌可能是整牌型（连对）
            count = Counter(poker)
            if (
                "jo" in count.keys()
                and "Jo" in count.keys()
                and count["jo"] == 2
                and count["Jo"] == 2
                and len(poker) == 4
            ):
                return "tractor"
            elif "jo" in count.keys() or "Jo" in count.keys():  # 排除大小王
                return "suspect"
            for v in count.values():  # 每组两张
                if v != 2:
                    return "suspect"
            pointpos = []
            suit = list(count.keys())[0][0]  # 花色相同
            for k in count.keys():
                if k[0] != suit or k[1] == level:  # 排除级牌
                    return "suspect"
                pointpos.append(self.point_order.index(k[1]))  # 点数在大小上连续
            pointpos.sort()
            for i in range(len(pointpos) - 1):
                if pointpos[i + 1] - pointpos[i] != 1:
                    return "suspect"
            return "tractor"  # 说明是拖拉机

        return "suspect"

    def _checkBigger(self, poker, currplayer):
        own = self.player_decks
        level = self.level
        major = self.major
        tyPoker = self._checkPokerType(poker, currplayer)
        poker = [self._id2name(p) for p in poker]
        assert tyPoker != "suspect", "Type 'throw' should contain common types"
        own_pok = [[self._id2name(num) for num in hold] for hold in own]
        if poker[0] in self.Major:  # 主牌型应用主牌压
            for i in range(len(own_pok)):
                if i == currplayer:
                    continue
                hold = own_pok[i]
                major_pok = [pok for pok in hold if pok in self.Major]
                count = Counter(major_pok)
                if len(poker) <= 2:
                    if (
                        poker[0][1] == level and poker[0][0] != major
                    ):  # 含有副级牌要单算
                        if major == "n":  # 无主
                            for k, v in count.items():
                                if (k == "jo" or k == "Jo") and v >= len(poker):
                                    return True
                        else:
                            for k, v in count.items():
                                if (
                                    k == "jo" or k == "Jo" or k == major + level
                                ) and v >= len(poker):
                                    return True
                    else:
                        for k, v in count.items():
                            if self.Major.index(k) > self.Major.index(
                                poker[0]
                            ) and v >= len(poker):
                                return True
                else:  # 拖拉机
                    if "jo" in poker:  # 必定是大小王连对
                        return False  # 不可能被压
                    if (
                        len(poker) == 4
                        and "jo" in count.keys()
                        and "Jo" in count.keys()
                    ):
                        if count["jo"] == 2 and count["Jo"] == 2:  # 大小王连对必压
                            return True
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if (
                                k != "jo"
                                and k != "Jo"
                                and k[1] != level
                                and self.point_order.index(k[1])
                                > self.point_order.index(poker[-1][1])
                            ):  # 大小王和级牌当然不会参与拖拉机
                                pos.append(self.point_order.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos) - 1):
                            if pos[i + 1] - pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker) / 2:
                                    return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False
        else:  # 副牌甩牌
            suit = poker[0][0]
            for i in range(len(own_pok)):
                if i == currplayer:
                    continue
                hold = own_pok[i]
                suit_pok = [pok for pok in hold if pok[0] == suit and pok[1] != level]
                count = Counter(suit_pok)
                if len(poker) <= 2:
                    for k, v in count.items():
                        if self.point_order.index(k[1]) > self.point_order.index(
                            poker[0][1]
                        ) and v >= len(poker):
                            return True
                else:
                    pos = []
                    for k, v in count.items():
                        if v == 2:
                            if self.point_order.index(k[1]) > self.point_order.index(
                                poker[-1][1]
                            ):
                                pos.append(self.point_order.index(k[1]))
                    if len(pos) >= 2:
                        pos.sort()
                        tmp = 0
                        suc_flag = False
                        for i in range(len(pos) - 1):
                            if pos[i + 1] - pos[i] == 1:
                                if not suc_flag:
                                    tmp = 2
                                    suc_flag = True
                                else:
                                    tmp += 1
                                if tmp >= len(poker) / 2:
                                    return True
                            elif suc_flag:
                                tmp = 0
                                suc_flag = False

        return False

    def _checkThrow(self, poker, currplayer, check=False):
        own = self.player_decks
        level = self.level
        major = self.major
        ilcnt = 0
        pok = [self._id2name(p) for p in poker]
        outpok = []
        failpok = []
        count = Counter(pok)
        if check:
            if list(count.keys())[0] in self.Major:  # 如果是主牌甩牌
                for p in count.keys():
                    if p not in self.Major:
                        self._raise_error(currplayer, "INVALID_POKERTYPE")
            else:  # 是副牌
                suit = list(count.keys())[0][0]  # 花色相同
                for k in count.keys():
                    if k[0] != suit:
                        self._raise_error(currplayer, "INVALID_POKERTYPE")
        pos = []
        tractor = []
        if count.get("jo") == 2 and count.get("Jo") == 2:
            tractor.append(["jo", "jo", "Jo", "Jo"])
            del count["jo"], count["Jo"]
        suit = ""
        for k, v in count.items():
            if v == 2:
                if (
                    k != "jo" and k != "Jo" and k[1] != level
                ):  # 大小王和级牌当然不会参与拖拉机
                    pos.append(self.point_order.index(k[1]))
                    suit = k[0]
        if len(pos) >= 2:
            pos.sort()
            tmp = []
            suc_flag = False
            for i in range(len(pos) - 1):
                if pos[i + 1] - pos[i] == 1:
                    if not suc_flag:
                        tmp = [
                            suit + self.point_order[pos[i]],
                            suit + self.point_order[pos[i]],
                            suit + self.point_order[pos[i + 1]],
                            suit + self.point_order[pos[i + 1]],
                        ]
                        del count[suit + self.point_order[pos[i]]]
                        del count[
                            suit + self.point_order[pos[i + 1]]
                        ]  # 已计入拖拉机的，从牌组中删去
                        suc_flag = True
                    else:
                        tmp.extend(
                            [
                                suit + self.point_order[pos[i + 1]],
                                suit + self.point_order[pos[i + 1]],
                            ]
                        )
                        del count[suit + self.point_order[pos[i + 1]]]
                elif suc_flag:
                    tractor.append(tmp)
                    suc_flag = False
            if suc_flag:
                tractor.append(tmp)
        for k, v in count.items():
            outpok.append([k for i in range(v)])
        outpok.extend(tractor)

        if check:
            for poktype in outpok:
                if self._checkBigger(
                    self._name2id_seq(poktype, list(range(108))), currplayer
                ):  # 甩牌失败
                    ilcnt += len(poktype)
                    failpok.append(poktype)

        if ilcnt > 0:
            finalpok = []
            kmin = ""
            for poktype in failpok:
                getmark = poktype[-1]
                if kmin == "":
                    finalpok = poktype
                    kmin = getmark
                elif kmin in self.Major:  # 主牌甩牌
                    if self.Major.index(getmark) < self.Major.index(kmin):
                        finalpok = poktype
                        kmin = getmark
                else:  # 副牌甩牌
                    if self.point_order.index(getmark[1]) < self.point_order.index(
                        kmin[1]
                    ):
                        finalpok = poktype
                        kmin = getmark
            finalpok = [[finalpok[0]]]
        else:
            finalpok = outpok

        return finalpok, ilcnt

    def _checkRes(self, poker, own):  # poker: list[int]
        level = self.level
        pok = [self._id2name(p) for p in poker]
        own_pok = [self._id2name(p) for p in own]
        if pok[0] in self.Major:
            major_pok = [pok for pok in own_pok if pok in self.Major]
            count = Counter(major_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker):
                        return True
            else:  # 拖拉机
                pos = []
                for k, v in count.items():
                    if v == 2:
                        if (
                            k != "jo" and k != "Jo" and k[1] != level
                        ):  # 大小王和级牌当然不会参与拖拉机
                            pos.append(self.point_order.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos) - 1):
                        if pos[i + 1] - pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker) / 2:
                                return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        else:
            suit = pok[0][0]
            suit_pok = [pok for pok in own_pok if pok[0] == suit and pok[1] != level]
            count = Counter(suit_pok)
            if len(poker) <= 2:
                for v in count.values():
                    if v >= len(poker):
                        return True
            else:
                pos = []
                for k, v in count.items():
                    if v == 2:
                        pos.append(self.point_order.index(k[1]))
                if len(pos) >= 2:
                    pos.sort()
                    tmp = 0
                    suc_flag = False
                    for i in range(len(pos) - 1):
                        if pos[i + 1] - pos[i] == 1:
                            if not suc_flag:
                                tmp = 2
                                suc_flag = True
                            else:
                                tmp += 1
                            if tmp >= len(poker) / 2:
                                return True
                        elif suc_flag:
                            tmp = 0
                            suc_flag = False
        return False

    def _checkLegalMove(
        self, poker, currplayer
    ):  # own: All players' hold before this move
        level = self.level
        major = self.major
        own = self.player_decks
        banker = self.banker_pos
        history = self.history
        pok = [self._id2name(p) for p in poker]
        hist = [[self._id2name(p) for p in move] for move in history]
        outpok = pok
        own_pok = [self._id2name(p) for p in own[currplayer]]
        if len(history) == 0 or len(history) == 4:  # The first move in a round
            typoker = self._checkPokerType(poker, currplayer)
            if typoker == "suspect":
                outpok_s, ilcnt = self._checkThrow(poker, currplayer, True)
                if ilcnt > 0:
                    self._punish(currplayer, ilcnt * 10)
                outpok = [
                    p for poktype in outpok_s for p in poktype
                ]  # 符合交互模式，把甩牌展开
        else:
            tyfirst = self._checkPokerType(history[0], currplayer)
            if len(poker) != len(history[0]):
                self._raise_error(currplayer, "ILLEGAL_MOVE")
            lead = hist[0][0]
            category = lambda c: "trump" if c in self.Major else c[0]
            held_suit = [c for c in own_pok if category(c) == category(lead)]
            played_suit = [c for c in pok if category(c) == category(lead)]
            if len(played_suit) != min(len(held_suit), len(poker)):
                self._raise_error(currplayer, "MUST_FOLLOW_SUIT")
            if len(held_suit) >= len(poker):
                pairs = lambda cards: sum(n // 2 for n in Counter(cards).values())
                if pairs(pok) < min(pairs(held_suit), pairs(hist[0])):
                    self._raise_error(currplayer, "MUST_FOLLOW_PAIRS")
            if tyfirst == "suspect":  # 这里own不一样了，但是可以不需要check
                outhis, ilcnt = self._checkThrow(history[0], currplayer, check=False)
                flathis = [p for poktype in outhis for p in poktype]
                if outhis[0][0] in self.Major:
                    major_pok = [p for p in pok if p in self.Major]
                    if len(major_pok) != len(
                        poker
                    ):  # 这种情况下，同花(主牌)必须已经贴完
                        major_hold = [p for p in own_pok if p in self.Major]
                        if len(major_pok) != len(major_hold):
                            self._raise_error(currplayer, "ILLEGAL_MOVE")
                    else:  # 全是主牌
                        outhis.sort(
                            key=lambda x: len(x), reverse=True
                        )  # 牌型从大到小来看
                        major_hold = [p for p in own_pok if p in self.Major]
                        matching = True
                        if (
                            self._checkPokerType(
                                self._name2id_seq(outhis[0], list(range(108))),
                                currplayer,
                            )
                            == "tractor"
                        ):  # 拖拉机来喽
                            divider, _ = self._checkThrow(
                                poker, currplayer, check=False
                            )
                            divider.sort(key=lambda x: len(x), reverse=True)
                            dividcnt = [len(x) for x in divider]
                            own_divide, r = self._checkThrow(
                                self._name2id_seq(major_hold, list(range(108))),
                                currplayer,
                                check=False,
                            )
                            own_divide.sort(key=lambda x: len(x), reverse=True)
                            own_cnt = [len(x) for x in own_divide]
                            for (
                                poktype
                            ) in (
                                outhis
                            ):  # 可以使用这种方法的原因在于同一组花色/主牌可组成的牌型数量太少，不会出现多解
                                if dividcnt[0] >= len(poktype):
                                    dividcnt[0] -= len(poktype)
                                    dividcnt.sort(reverse=True)
                                else:
                                    matching = False
                                    break
                            if not matching:  # 不匹配，看手牌是否存在应手
                                res_ex = True
                                for chtype in outhis:
                                    if own_cnt[0] >= len(chtype):
                                        own_cnt[0] -= len(chtype)
                                        own_cnt.sort(reverse=True)
                                    else:
                                        res_ex = False
                                        break
                                if res_ex:  # 存在应手，说明贴牌不当
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")
                                else:  # 存在应手，继续检查
                                    pair_own = sum(
                                        [len(x) for x in own_divide if len(x) >= 2]
                                    )
                                    pair_his = sum(
                                        [len(x) for x in outhis if len(x) >= 2]
                                    )
                                    pair_pok = sum(
                                        [len(x) for x in divider if len(x) >= 2]
                                    )
                                    if pair_pok < min(pair_own, pair_his):
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                else:
                    suit = hist[0][0][0]
                    suit_pok = [p for p in pok if p not in self.Major and p[0] == suit]
                    if len(suit_pok) != len(
                        poker
                    ):  # 这种情况下，同花(主牌)必须已经贴完
                        suit_hold = [
                            p for p in own_pok if p not in self.Major and p[0] == suit
                        ]
                        if len(suit_pok) != len(suit_hold):
                            self._raise_error(currplayer, "ILLEGAL_MOVE")
                    else:
                        outhis.sort(
                            key=lambda x: len(x), reverse=True
                        )  # 牌型从大到小来看
                        suit_hold = [
                            p for p in own_pok if p not in self.Major and p[0] == suit
                        ]
                        matching = True
                        if (
                            self._checkPokerType(
                                self._name2id_seq(outhis[0], list(range(108))),
                                currplayer,
                            )
                            == "tractor"
                        ):  # 拖拉机来喽
                            divider, _ = self._checkThrow(
                                poker, currplayer, check=False
                            )
                            divider.sort(key=lambda x: len(x), reverse=True)
                            dividcnt = [len(x) for x in divider]
                            own_divide, r = self._checkThrow(
                                self._name2id_seq(suit_hold, list(range(108))),
                                currplayer,
                                check=False,
                            )
                            own_divide.sort(key=lambda x: len(x), reverse=True)
                            own_cnt = [len(x) for x in own_divide]
                            for (
                                poktype
                            ) in (
                                outhis
                            ):  # 可以使用这种方法的原因在于同一组花色/主牌可组成的牌型数量太少，不会出现多解
                                if dividcnt[0] >= len(poktype):
                                    dividcnt[0] -= len(poktype)
                                    dividcnt.sort(reverse=True)
                                else:
                                    matching = False
                                    break
                            if not matching:  # 不匹配，看手牌是否存在应手
                                res_ex = True
                                for chtype in outhis:
                                    if own_cnt[0] >= len(chtype):
                                        own_cnt[0] -= len(chtype)
                                        own_cnt.sort(reverse=True)
                                    else:
                                        res_ex = False
                                        break
                                if res_ex:  # 存在应手，说明贴牌不当
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")
                                else:  # 存在应手，继续检查
                                    pair_own = sum(
                                        [len(x) for x in own_divide if len(x) >= 2]
                                    )
                                    pair_his = sum(
                                        [len(x) for x in outhis if len(x) >= 2]
                                    )
                                    pair_pok = sum(
                                        [len(x) for x in divider if len(x) >= 2]
                                    )
                                    if pair_pok < min(pair_own, pair_his):
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")

            else:  # 常规牌型
                if self._checkRes(history[0], own[currplayer]):  # (1) 有应手但贴牌或毙
                    if self._checkPokerType(poker, currplayer) != tyfirst:
                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                    if hist[0][0] in self.Major and pok[0] not in self.Major:
                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                    if hist[0][0] not in self.Major and (
                        pok[0] in self.Major or pok[0][0] != hist[0][0][0]
                    ):
                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                elif (
                    self._checkPokerType(poker, currplayer) != tyfirst
                ):  # (2) 贴牌不当: 有同花不贴完/同花色不跟整牌型
                    own_pok = [self._id2name(p) for p in own[currplayer]]
                    if hist[0][0] in self.Major:
                        major_pok = [p for p in pok if p in self.Major]
                        major_hold = [p for p in own_pok if p in self.Major]
                        if len(major_pok) != len(
                            poker
                        ):  # 这种情况下，同花(主牌)必须已经贴完
                            if len(major_pok) != len(major_hold):
                                self._raise_error(currplayer, "ILLEGAL_MOVE")
                        else:  # 完全是主牌
                            count = Counter(major_hold)
                            if tyfirst == "pair":
                                for v in count.values():
                                    if v == 2:
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == "tractor":
                                trpairs = len(history[0]) / 2
                                pkcount = Counter(pok)
                                pkpairs = 0
                                hdpairs = 0
                                for v in pkcount.values():
                                    if v >= 2:
                                        pkpairs += 1
                                for v in count.values():
                                    if v >= 2:
                                        hdpairs += 1
                                if (
                                    pkpairs < trpairs and pkpairs < hdpairs
                                ):  # 并不是所有对子都用上了
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")

                    else:
                        suit = hist[0][0][0]
                        suit_pok = [
                            p for p in pok if p[0] == suit and p not in self.Major
                        ]
                        suit_hold = [
                            p for p in own_pok if p[0] == suit and p not in self.Major
                        ]
                        if len(suit_pok) != len(poker):
                            if len(suit_pok) != len(suit_hold):
                                self._raise_error(currplayer, "ILLEGAL_MOVE")
                        else:  # 完全是同种花色
                            count = Counter(suit_hold)
                            if tyfirst == "pair":
                                for v in count.values():
                                    if v == 2:
                                        self._raise_error(currplayer, "ILLEGAL_MOVE")
                            elif tyfirst == "tractor":
                                trpairs = len(history[0]) / 2
                                pkcount = Counter(pok)
                                pkpairs = 0
                                hdpairs = 0
                                for v in pkcount.values():
                                    if v >= 2:
                                        pkpairs += 1
                                for v in count.values():
                                    if v >= 2:
                                        hdpairs += 1
                                if (
                                    pkpairs < trpairs and pkpairs < hdpairs
                                ):  # 并不是所有对子都用上了
                                    self._raise_error(currplayer, "ILLEGAL_MOVE")

        return outpok

    def _checkWinner(self, currplayer):
        """Resolve a trick independently of physical card ordering; equal ranks retain the lead."""
        cards = [[self._id2name(c) for c in move] for move in self.history]

        def category(card):
            return "trump" if card in self.Major else card[0]

        def power(card):
            if card == "Jo":
                return 100
            if card == "jo":
                return 99
            if card[1] == self.level:
                return 98 if self.major != "n" and card[0] == self.major else 97
            return self.point_order.index(card[1])

        lead_type = self._checkPokerType(self.history[0], currplayer)
        lead_parts, _ = self._checkThrow(self.history[0], currplayer, check=False)
        wanted = sorted([len(p) for p in lead_parts], reverse=True)
        winner, best = 0, None
        for i, move in enumerate(cards):
            categories = {category(c) for c in move}
            if len(categories) != 1:
                continue
            cat = next(iter(categories))
            if cat not in (category(cards[0][0]), "trump"):
                continue
            typ = self._checkPokerType(self.history[i], currplayer)
            if lead_type != "suspect" and typ != lead_type:
                continue
            parts, _ = self._checkThrow(self.history[i], currplayer, check=False)
            if lead_type == "suspect":
                sizes = sorted([len(p) for p in parts], reverse=True)
                fits = True
                for size in wanted:
                    if not sizes or sizes[0] < size:
                        fits = False
                        break
                    sizes[0] -= size
                    sizes.sort(reverse=True)
                if not fits:
                    continue
                largest = wanted[0]
                strength = max(
                    max(power(c) for c in part)
                    for part in parts
                    if len(part) >= largest
                )
            else:
                strength = max(power(c) for c in move)
            score = (int(cat == "trump"), strength)
            if best is None or score > best:
                winner, best = i, score
        player = (currplayer - 3 + winner) % 4
        points = sum(
            5 if c[1] == "5" else 10 if c[1] in ("0", "K") else 0
            for move in cards
            for c in move
        )
        self._reward(player, points)
        return player

    def _reward(self, player, points):
        if (player - self.banker_pos) % 2 != 0:  # farmer getting points
            self.score += points
        self.reward = {}
        for i in range(4):
            if (i - player) % 2 == 0:
                self.reward[self.agent_names[i]] = points
            else:
                self.reward[self.agent_names[i]] = -points

    def _punish(self, player, points):
        if (player - self.banker_pos) % 2 != 0:
            self.score -= points
        else:
            self.score += points

    def action_intpt(self, action, player):
        """
        interpreting action(cardname) to response(dick{'player': int, 'action': list[int]})
        action: list[str(cardnames)]
        """
        player_deck = self.player_decks[player]
        action = self._name2id_seq(action, player_deck)
        return {"player": player, "action": action}
