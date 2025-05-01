Okay, here is the mixed set of problems covering combinations and permutations, formatted as Markdown text. For each problem, you first need to identify whether order matters, if repetition is allowed, and if any items are non-distinct, then apply the appropriate formula or reasoning. Solutions follow each problem.

Generic Definitions:
 * n: Generally represents the size or scope of the set you are choosing from.
   * This could be the total number of distinct items available.
   * It could also represent the number of distinct categories or types of items available when replacement is allowed or items are non-distinct.
 * k: Generally represents the number of items being selected, arranged, or positions being filled.
   * This is the size of the subgroup or the length of the sequence you are forming.

**Key Formulas Reminder:**
* Permutation without Replacement: $P(n, k) = \frac{n!}{(n-k)!}$
* Permutation with Replacement: $n^k$
* Combination without Replacement: $C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!}$
* Combination with Replacement: $C(n+k-1, k) = \binom{n+k-1}{k} = \frac{(n+k-1)!}{k!(n-1)!}$
* Permutations with Non-Distinct Items: $\frac{n!}{n_1! n_2! \dots n_k!}$
* Circular Permutations: $(n-1)!$
* Probability: $P(\text{Event}) = \frac{\text{Number of Favorable Outcomes}}{\text{Total Number of Possible Outcomes}}$

---

## Mixed Problems

1.  **Problem:** How many ways can 6 students line up for a photo?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(6, 6) = 6! = 720$

2.  **Problem:** A bag contains many red, blue, and green marbles. How many ways can you select 5 marbles?
    * *Identify:* Order doesn't matter, replacement allowed (many of each color). Combination with replacement.
    * **Solution:** $n=3$ colors, $k=5$ marbles. $C(3+5-1, 5) = C(7, 5) = \frac{7!}{5!2!} = 21$

3.  **Problem:** How many different 4-digit PINs are possible using digits 0-9 if digits can be repeated?
    * *Identify:* Order matters, replacement allowed. Permutation with replacement.
    * **Solution:** $10^4 = 10,000$

4.  **Problem:** How many ways can you arrange the letters in the word "BOOK"?
    * *Identify:* Order matters, non-distinct items (O repeats). Permutation with non-distinct items.
    * **Solution:** $n=4$, $n_O=2$. $\frac{4!}{2!} = \frac{24}{2} = 12$

5.  **Problem:** A committee of 4 people is to be selected from a group of 10. How many different committees are possible?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(10, 4) = \frac{10!}{4!6!} = \frac{10 \times 9 \times 8 \times 7}{4 \times 3 \times 2 \times 1} = 210$

6.  **Problem:** How many 5-letter "words" can be formed using the English alphabet (26 letters) if repetition is allowed?
    * *Identify:* Order matters, replacement allowed. Permutation with replacement.
    * **Solution:** $26^5 = 11,881,376$

7.  **Problem (Complex):** Calculate the probability of getting a Full House (three cards of one rank, two cards of another rank) in a 5-card poker hand from a standard 52-card deck.
    * *Identify:* Probability using combinations. Need favorable outcomes / total outcomes.
    * **Solution:**
        * Choose rank for three cards: $C(13, 1) = 13$. Choose 3 suits: $C(4, 3) = 4$.
        * Choose rank for two cards: $C(12, 1) = 12$. Choose 2 suits: $C(4, 2) = 6$.
        * Favorable outcomes = $13 \times 4 \times 12 \times 6 = 3744$.
        * Total possible 5-card hands = $C(52, 5) = 2,598,960$.
        * Probability = $\frac{3744}{2,598,960} \approx 0.00144$ (or $6/4165$).

8.  **Problem:** How many ways can you choose 2 books to read from a list of 6?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(6, 2) = \frac{6!}{2!4!} = \frac{6 \times 5}{2 \times 1} = 15$

9.  **Problem:** How many ways can 5 different books be arranged on a shelf?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(5, 5) = 5! = 120$

10. **Problem:** How many different binary strings (sequences of 0s and 1s) of length 7 are possible?
    * *Identify:* Order matters, replacement allowed (0s and 1s). Permutation with replacement.
    * **Solution:** $n=2, k=7$. $2^7 = 128$

11. **Problem:** How many ways to choose 2 appetizers from a menu of 9?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(9, 2) = \frac{9!}{2!7!} = \frac{9 \times 8}{2 \times 1} = 36$

12. **Problem:** How many ways can the top 5 finishers be ordered in a race with 12 competitors?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(12, 5) = \frac{12!}{(12-5)!} = \frac{12!}{7!} = 12 \times 11 \times 10 \times 9 \times 8 = 95,040$

13. **Problem:** A bag contains 7 different colored marbles. How many ways can you pick 3 marbles?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(7, 3) = \frac{7!}{3!(7-3)!} = \frac{7!}{3!4!} = \frac{7 \times 6 \times 5}{3 \times 2 \times 1} = 35$

14. **Problem (Complex):** How many ways can the letters of the word "COMPUTER" be arranged so that the vowels (O, U, E) are never all together?
    * *Identify:* Complementary counting using permutations. Total arrangements minus arrangements where vowels ARE together.
    * **Solution:**
        * Total arrangements of "COMPUTER" (8 distinct letters) = $8! = 40,320$.
        * Treat vowels (OUE) as one block. Arrange {C,M,P,T,R, (OUE)}: $6! = 720$ ways.
        * Arrange vowels within the block: $3! = 6$ ways.
        * Arrangements with vowels together = $6! \times 3! = 720 \times 6 = 4320$.
        * Arrangements with vowels not all together = $40,320 - 4320 = 36,000$.

15. **Problem:** Selecting 3 toppings for a pizza from 8 available toppings.
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(8, 3) = \frac{8!}{3!(8-3)!} = \frac{8!}{3!5!} = \frac{8 \times 7 \times 6}{3 \times 2 \times 1} = 56$

16. **Problem:** How many ways can you select and arrange 2 paintings from a collection of 9?
    * *Identify:* Order matters (arrange), no replacement. Permutation without replacement.
    * **Solution:** $P(9, 2) = \frac{9!}{(9-2)!} = \frac{9!}{7!} = 9 \times 8 = 72$

17. **Problem:** Choosing 6 lottery numbers from 49.
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(49, 6) = \frac{49!}{6!(49-6)!} = \frac{49!}{6!43!} = 13,983,816$

18. **Problem:** A car license plate consists of 3 letters followed by 3 digits. Repetition is allowed. How many different plates are possible?
    * *Identify:* Order matters, replacement allowed. Use multiplication principle for independent parts.
    * **Solution:** $26^3 \times 10^3 = 17,576 \times 1,000 = 17,576,000$

19. **Problem:** How many distinct arrangements are there for the letters in "BANANA"?
    * *Identify:* Order matters, non-distinct items (A, N repeat). Permutation with non-distinct items.
    * **Solution:** $n=6$, $n_A=3, n_N=2$. $\frac{6!}{1! 3! 2!} = \frac{720}{1 \times 6 \times 2} = 60$

20. **Problem:** How many ways can you choose 4 scoops of ice cream from 10 available flavors, if you can have multiple scoops of the same flavor?
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.
    * **Solution:** $n=10$ flavors, $k=4$ scoops. $C(10+4-1, 4) = C(13, 4) = \frac{13!}{4!(13-4)!} = \frac{13!}{4!9!} = 715$

21. **Problem:** In a race with 8 runners, how many ways can the gold, silver, and bronze medals be awarded?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(8, 3) = \frac{8!}{(8-3)!} = \frac{8!}{5!} = 8 \times 7 \times 6 = 336$

22. **Problem:** How many ways can you select 4 volunteers from a group of 15?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(15, 4) = \frac{15!}{4!(15-4)!} = \frac{15!}{4!11!} = \frac{15 \times 14 \times 13 \times 12}{4 \times 3 \times 2 \times 1} = 1365$

23. **Problem:** A quiz has 10 multiple-choice questions, each with 4 options (A, B, C, D). How many ways can a student answer the quiz?
    * *Identify:* Order matters (sequence of answers), replacement allowed (same option can be chosen for different questions). Permutation with replacement.
    * **Solution:** $n=4, k=10$. $4^{10} = 1,048,576$

24. **Problem:** How many ways can you choose 3 fruits from a basket containing 5 different fruits?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(5, 3) = \frac{5!}{3!(5-3)!} = \frac{5!}{3!2!} = \frac{5 \times 4}{2 \times 1} = 10$

25. **Problem (Complex):** A committee of 5 people is to be formed from 6 physicists, 5 chemists, and 4 mathematicians. How many ways can the committee be formed if it must contain exactly 2 physicists and at least 2 chemists?
    * *Identify:* Combinations with constraints, requiring cases.
    * **Solution:**
        * Choose 2 Physicists: $C(6, 2) = 15$. Need 3 more members.
        * Case 1: 2 Chemists, 1 Mathematician. Choose Chemists: $C(5, 2) = 10$. Choose Mathematician: $C(4, 1) = 4$. Ways = $15 \times 10 \times 4 = 600$.
        * Case 2: 3 Chemists, 0 Mathematicians. Choose Chemists: $C(5, 3) = 10$. Choose Mathematician: $C(4, 0) = 1$. Ways = $15 \times 10 \times 1 = 150$.
        * Total ways = $600 + 150 = 750$.

26. **Problem:** How many ways can you arrange 3 red, 2 blue, and 1 green flag on a flagpole?
    * *Identify:* Order matters, non-distinct items (flags of same color). Permutation with non-distinct items.
    * **Solution:** $n=6$, $n_R=3, n_B=2, n_G=1$. $\frac{6!}{3! 2! 1!} = \frac{720}{6 \times 2 \times 1} = 60$

27. **Problem:** How many ways can you arrange the letters A, B, C?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(3, 3) = 3! = 6$

28. **Problem:** How many ways can a student choose 8 questions to answer from an exam paper containing 10 questions?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(10, 8) = C(10, 2) = \frac{10!}{8!(10-8)!} = \frac{10!}{8!2!} = \frac{10 \times 9}{2 \times 1} = 45$

29. **Problem:** How many possible outcomes are there if you flip a coin 5 times?
    * *Identify:* Order matters (sequence of flips), replacement allowed (H/T on each flip). Permutation with replacement.
    * **Solution:** $n=2, k=5$. $2^5 = 32$

30. **Problem:** How many ways to choose a lunch combo: 1 sandwich from 5 options, 1 side from 4 options, 1 drink from 3 options?
    * *Identify:* Multiplication principle for independent choices.
    * **Solution:** $5 \times 4 \times 3 = 60$

31. **Problem:** Distributing 7 identical candies among 3 children (where some children might get none).
    * *Identify:* Order doesn't matter (identical candies), replacement allowed (child can get >1). Combination with replacement (Stars and Bars).
    * **Solution:** $n=3$ children, $k=7$ candies. $C(3+7-1, 7) = C(9, 7) = C(9, 2) = \frac{9!}{7!(9-7)!} = \frac{9!}{7!2!} = 36$

32. **Problem:** How many ways can you choose 10 balls from a bag containing 12 distinct balls?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(12, 10) = C(12, 2) = \frac{12!}{10!(12-10)!} = \frac{12!}{10!2!} = \frac{12 \times 11}{2 \times 1} = 66$

33. **Problem:** How many 3-digit numbers can be formed using the digits 1, 2, 3, 4, 5 without repetition?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(5, 3) = \frac{5!}{(5-3)!} = \frac{5!}{2!} = 5 \times 4 \times 3 = 60$

34. **Problem:** How many different signals can be made using 4 white flags, 3 blue flags, and 2 red flags arranged vertically?
    * *Identify:* Order matters, non-distinct items. Permutation with non-distinct items.
    * **Solution:** $n=9$, $n_W=4, n_B=3, n_R=2$. $\frac{9!}{4! 3! 2!} = \frac{362,880}{24 \times 6 \times 2} = 1260$

35. **Problem:** A standard combination lock has numbers 0-39 (40 numbers). How many 3-number combinations are possible if numbers can be repeated? (Note: mathematically this is a permutation as order matters).
    * *Identify:* Order matters, replacement allowed. Permutation with replacement.
    * **Solution:** $n=40, k=3$. $40^3 = 64,000$

36. **Problem:** Selecting a starting basketball team of 5 players from a squad of 12.
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(12, 5) = \frac{12!}{5!(12-5)!} = \frac{12!}{5!7!} = \frac{12 \times 11 \times 10 \times 9 \times 8}{5 \times 4 \times 3 \times 2 \times 1} = 792$

37. **Problem (Complex):** In how many ways can 6 people be seated around a circular table if two specific people must sit together?
    * *Identify:* Circular permutation with constraint.
    * **Solution:**
        * Treat the two specific people (A, B) as one unit.
        * Arrange this unit and the other 4 people around the circle: $(5 \text{ units} - 1)! = 4! = 24$ ways.
        * The two specific people can arrange themselves within their unit (A, B or B, A): $2! = 2$ ways.
        * Total ways = $4! \times 2! = 24 \times 2 = 48$.

38. **Problem:** How many ways can you select 3 T-shirts from a store that offers them in 7 different colors (assuming ample stock)?
    * *Identify:* Order doesn't matter, replacement allowed (can choose same color). Combination with replacement.
    * **Solution:** $n=7$ colors, $k=3$ shirts. $C(7+3-1, 3) = C(9, 3) = \frac{9!}{3!(9-3)!} = \frac{9!}{3!6!} = 84$

39. **Problem:** How many different 7-digit numbers can be formed using the digits 3, 3, 3, 4, 4, 5, 5?
    * *Identify:* Order matters, non-distinct items. Permutation with non-distinct items.
    * **Solution:** $n=7$, $n_3=3, n_4=2, n_5=2$. $\frac{7!}{3! 2! 2!} = \frac{5040}{6 \times 2 \times 2} = 210$

40. **Problem:** How many ways can 4 boys and 3 girls sit in a row if all the boys must sit together and all the girls must sit together?
    * *Identify:* Permutations with grouping constraint. Treat groups as blocks.
    * **Solution:** Treat boys as block B, girls as block G. Arrange blocks (BG or GB): 2 ways. Arrange boys within B: 4! ways. Arrange girls within G: 3! ways. Total = $2 \times 4! \times 3! = 2 \times 24 \times 6 = 288$.

41. **Problem:** How many ways to choose a team of 4 from 10 people if two specific people refuse to be on the team together?
    * *Identify:* Complementary counting using combinations. Total ways minus ways they ARE together.
    * **Solution:** Total ways: $C(10, 4) = 210$. Ways with the two specific people together (choose 2 more from remaining 8): $C(8, 2) = 28$. Ways they are not together: $210 - 28 = 182$.

42. **Problem:** How many 4-letter arrangements can be made from the letters in "MATH"?
    * *Identify:* Order matters, no replacement (distinct letters). Permutation without replacement.
    * **Solution:** $P(4, 4) = 4! = 24$

43. **Problem:** Selecting 2 drinks from a machine offering 6 choices, repetition allowed.
    * *Identify:* Order doesn't matter (final selection), replacement allowed. Combination with replacement.
    * **Solution:** $n=6$ choices, $k=2$ drinks. $C(6+2-1, 2) = C(7, 2) = \frac{7!}{2!(7-2)!} = \frac{7!}{2!5!} = 21$

44. **Problem:** How many ways can the letters in "STATISTICS" be arranged?
    * *Identify:* Order matters, non-distinct items. Permutation with non-distinct items.
    * **Solution:** $n=10$, $n_S=3, n_T=3, n_I=2$. $\frac{10!}{3! 3! 1! 2! 1!} = \frac{3,628,800}{6 \times 6 \times 1 \times 2 \times 1} = 50,400$

45. **Problem:** From a group of 7 people, how many ways can a chairperson and a secretary be selected?
    * *Identify:* Order matters (distinct roles), no replacement. Permutation without replacement.
    * **Solution:** $P(7, 2) = \frac{7!}{(7-2)!} = \frac{7!}{5!} = 7 \times 6 = 42$

46. **Problem (Complex):** How many non-negative integer solutions are there to $x_1 + x_2 + x_3 = 15$ such that $x_1 \ge 1$, $x_2 \ge 2$, $x_3 \ge 0$?
    * *Identify:* Combination with replacement (Stars and Bars) after variable substitution for lower bounds.
    * **Solution:**
        * Let $y_1 = x_1 - 1 \ge 0$, $y_2 = x_2 - 2 \ge 0$, $y_3 = x_3 \ge 0$.
        * Substitute: $(y_1 + 1) + (y_2 + 2) + y_3 = 15 \implies y_1 + y_2 + y_3 = 12$.
        * Solve for $y_1, y_2, y_3 \ge 0$ using Stars and Bars: $n=3$ variables, $k=12$ units.
        * Ways = $C(n+k-1, k) = C(3+12-1, 12) = C(14, 12) = C(14, 2) = \frac{14 \times 13}{2} = 91$.

47. **Problem:** How many ways can you arrange 4 out of 6 different flower vases in a row?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(6, 4) = \frac{6!}{(6-4)!} = \frac{6!}{2!} = 6 \times 5 \times 4 \times 3 = 360$

48. **Problem:** Picking a team of 3 men and 2 women from 7 men and 5 women.
    * *Identify:* Two independent combinations without replacement. Use multiplication principle.
    * **Solution:** Choose men: $C(7, 3) = 35$. Choose women: $C(5, 2) = 10$. Total ways = $35 \times 10 = 350$.

49. **Problem:** A company has 15 employees. How many ways can they select a CEO, CFO, and COO?
    * *Identify:* Order matters (distinct roles), no replacement. Permutation without replacement.
    * **Solution:** $P(15, 3) = \frac{15!}{(15-3)!} = \frac{15!}{12!} = 15 \times 14 \times 13 = 2730$

50. **Problem:** Choosing 5 coins from a wallet containing pennies, nickels, dimes, and quarters (assume you have at least 5 of each).
    * *Identify:* Order doesn't matter, replacement allowed (can pick multiple of same type). Combination with replacement.
    * **Solution:** $n=4$ coin types, $k=5$ coins. $C(4+5-1, 5) = C(8, 5) = C(8, 3) = \frac{8!}{5!3!} = 56$

51. **Problem:** How many ways can 8 different cars be parked in 8 adjacent parking spots?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(8, 8) = 8! = 40,320$

52. **Problem:** How many ways can a group of 5 people be chosen from 8 if a particular person must be included?
    * *Identify:* Combination without replacement with constraint. Fix one person, choose the rest.
    * **Solution:** Person is fixed. Choose remaining 4 from remaining 7 people: $C(7, 4) = C(7, 3) = 35$.

53. **Problem:** How many 3-digit numbers can be formed using the digits 1, 2, 3, 4, 5 if repetition is allowed?
    * *Identify:* Order matters, replacement allowed. Permutation with replacement.
    * **Solution:** $n=5, k=3$. $5^3 = 125$

54. **Problem:** How many distinct ways can you arrange the letters in the word "MISSISSIPPI"?
    * *Identify:* Order matters, non-distinct items. Permutation with non-distinct items.
    * **Solution:** $n=11$, $n_I=4, n_S=4, n_P=2$. $\frac{11!}{1! 4! 4! 2!} = \frac{39,916,800}{1 \times 24 \times 24 \times 2} = 34,650$

55. **Problem:** How many ways can you choose 5 apples from a basket containing a large supply of Red Delicious, Granny Smith, and Fuji apples?
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.
    * **Solution:** $n=3$ types, $k=5$ apples. $C(3+5-1, 5) = C(7, 5) = C(7, 2) = 21$

56. **Problem:** From a class of 25 students, how many ways can a group of 5 be chosen for a project?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(25, 5) = \frac{25!}{5!(25-5)!} = \frac{25!}{5!20!} = \frac{25 \times 24 \times 23 \times 22 \times 21}{5 \times 4 \times 3 \times 2 \times 1} = 53,130$

57. **Problem:** How many ways can a student select answers for a 5-question quiz where each question has 3 choices (A, B, C)?
    * *Identify:* Order matters (sequence of answers), replacement allowed. Permutation with replacement.
    * **Solution:** $n=3, k=5$. $3^5 = 243$

58. **Problem (Complex):** How many ways are there to travel from the bottom-left corner (0,0) to the top-right corner (5,4) of a grid, moving only right (R) or up (U)?
    * *Identify:* Pathfinding on a grid is equivalent to arranging sequences of moves. Combination or Permutation with non-distinct items.
    * **Solution:** Need 5 Right moves and 4 Up moves, total 9 moves. Arrange the sequence RRRRRUUUU. Ways = $\frac{9!}{5!4!} = C(9, 5) = C(9, 4) = 126$.

59. **Problem:** How many ways can you choose 3 movies to watch from a list of 10?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(10, 3) = \frac{10!}{3!(10-3)!} = \frac{10!}{3!7!} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = 120$

60. **Problem:** Arrange 5 identical pennies and 3 identical dimes in a row.
    * *Identify:* Order matters, non-distinct items. Permutation with non-distinct items (or choose positions).
    * **Solution:** $n=8$, $n_P=5, n_D=3$. $\frac{8!}{5! 3!} = C(8, 3) = 56$

61. **Problem:** A club has 10 members. How many ways can a President and Vice-President be chosen?
    * *Identify:* Order matters (distinct roles), no replacement. Permutation without replacement.
    * **Solution:** $P(10, 2) = \frac{10!}{(10-2)!} = \frac{10!}{8!} = 10 \times 9 = 90$

62. **Problem:** How many ways can 10 different questions on a test be arranged?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(10, 10) = 10! = 3,628,800$

63. **Problem:** How many ways can you give 3 identical awards to a class of 12 students (a student can receive at most one award)?
    * *Identify:* Order doesn't matter (identical awards), no replacement. Combination without replacement.
    * **Solution:** Choose 3 students from 12: $C(12, 3) = \frac{12!}{3!(12-3)!} = \frac{12!}{3!9!} = 220$.

64. **Problem:** Picking 6 winning lottery numbers from 1 to 50, where order doesn't matter.
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(50, 6) = \frac{50!}{6!(50-6)!} = \frac{50!}{6!44!} = 15,890,700$

65. **Problem:** If you draw a card from a standard deck of 52, replace it, and draw again 4 times in total, how many sequences of cards are possible?
    * *Identify:* Order matters (sequence), replacement allowed. Permutation with replacement.
    * **Solution:** $n=52, k=4$. $52^4 = 7,311,616$

66. **Problem:** How many non-negative integer solutions are there to the equation $x_1 + x_2 + x_3 + x_4 = 10$?
    * *Identify:* Order doesn't matter (sum), replacement allowed (variables can be >1 or 0). Combination with replacement (Stars and Bars).
    * **Solution:** $n=4$ variables, $k=10$ units. $C(4+10-1, 10) = C(13, 10) = C(13, 3) = 286$.

67. **Problem:** How many ways can a group of 5 people be chosen from 8 if a particular person must be excluded?
    * *Identify:* Combination without replacement with constraint. Remove one person, then choose.
    * **Solution:** Exclude the person. Choose 5 from the remaining 7: $C(7, 5) = C(7, 2) = 21$.

68. **Problem:** How many ways to arrange the letters of "LEVEL"?
    * *Identify:* Order matters, non-distinct items (L, E repeat). Permutation with non-distinct items.
    * **Solution:** $n=5$, $n_L=2, n_E=2$. $\frac{5!}{2! 2!} = \frac{120}{4} = 30$

69. **Problem:** How many ways can 5 friends sit around a circular table?
    * *Identify:* Circular permutation.
    * **Solution:** $(n-1)! = (5-1)! = 4! = 24$

70. **Problem (Complex):** Calculate the probability of getting Two Pair (two cards of one rank, two cards of another rank, and one card of a third rank) in a 5-card poker hand.
    * *Identify:* Probability using combinations. Favorable outcomes / total outcomes.
    * **Solution:**
        * Choose the 2 ranks for the pairs: $C(13, 2) = 78$.
        * Choose 2 suits for the first pair: $C(4, 2) = 6$.
        * Choose 2 suits for the second pair: $C(4, 2) = 6$.
        * Choose the rank for the fifth card (from remaining 11 ranks): $C(11, 1) = 11$.
        * Choose the suit for the fifth card: $C(4, 1) = 4$.
        * Favorable outcomes = $78 \times 6 \times 6 \times 11 \times 4 = 123,552$.
        * Total possible 5-card hands = $C(52, 5) = 2,598,960$.
        * Probability = $\frac{123,552}{2,598,960} \approx 0.0475$ (or $198/4165$).

71. **Problem:** How many 2-letter codes can be formed from the letters P, Q, R, S without repeating letters?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(4, 2) = \frac{4!}{(4-2)!} = \frac{4!}{2!} = 4 \times 3 = 12$

72. **Problem:** How many ways can you select 5 marbles from a bag with a large supply of red, blue, green, and yellow marbles?
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.
    * **Solution:** $n=4$ colors, $k=5$ marbles. $C(4+5-1, 5) = C(8, 5) = C(8, 3) = 56$

73. **Problem:** How many ways can 5 math books and 3 physics books on a shelf if all books of the same subject must stay together?
    * *Identify:* Permutations with grouping constraint. Treat subjects as blocks. Assume books are distinct unless stated otherwise.
    * **Solution:** Blocks (M, P). Arrange blocks: 2! ways. Arrange Math books: 5! ways. Arrange Physics books: 3! ways. Total = $2! \times 5! \times 3! = 2 \times 120 \times 6 = 1440$.

74. **Problem:** A signal flag system uses 5 positions, and each position can be one of 4 colors. How many different signals can be made?
    * *Identify:* Order matters (position), replacement allowed (color can repeat). Permutation with replacement.
    * **Solution:** $n=4, k=5$. $4^5 = 1024$

75. **Problem:** How many ways can you choose 4 donuts from 6 types?
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.
    * **Solution:** $n=6$ types, $k=4$ donuts. $C(6+4-1, 4) = C(9, 4) = \frac{9!}{4!(9-4)!} = \frac{9!}{4!5!} = 126$

76. **Problem:** How many ways to arrange the letters in "ENGINEERING"?
    * *Identify:* Order matters, non-distinct items. Permutation with non-distinct items.
    * **Solution:** $n=11$, $n_E=3, n_N=3, n_G=2, n_I=2$. $\frac{11!}{3! 3! 2! 2! 1!} = \frac{39,916,800}{6 \times 6 \times 2 \times 2 \times 1} = 277,200$

77. **Problem:** How many ways can you choose an answer for each of 8 true/false questions?
    * *Identify:* Order matters (sequence of answers), replacement allowed. Permutation with replacement.
    * **Solution:** $n=2, k=8$. $2^8 = 256$

78. **Problem:** How many different 5-card hands can be dealt from a standard 52-card deck?
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.
    * **Solution:** $C(52, 5) = \frac{52!}{5!(52-5)!} = \frac{52!}{5!47!} = 2,598,960$

79. **Problem:** How many ways can a committee of 3 be formed from 6 men and 4 women if it must contain exactly 2 men?
    * *Identify:* Combination without replacement with constraint. Use multiplication principle.
    * **Solution:** Choose 2 men from 6: $C(6, 2)=15$. Choose 1 woman from 4: $C(4, 1)=4$. Total ways = $15 \times 4 = 60$.

80. **Problem:** Rolling 4 identical dice. How many distinct outcomes are possible (e.g., {1,2,3,4} is one outcome, {1,1,2,3} is another)?
    * *Identify:* Order doesn't matter (identical dice), replacement allowed (can roll same number). Combination with replacement.
    * **Solution:** $n=6$ faces, $k=4$ dice. $C(6+4-1, 4) = C(9, 4) = 126$.

81. **Problem:** How many ways can 10 identical balls into 4 distinct boxes.
    * *Identify:* Distributing identical items into distinct boxes. Combination with replacement (Stars and Bars).
    * **Solution:** $n=4$ boxes, $k=10$ balls. $C(4+10-1, 10) = C(13, 10) = C(13, 3) = 286$.

82. **Problem:** How many ways can 4 different objects using 3 available colors, if colors can be reused?
    * *Identify:* Assigning a color to each object. Order matters (Object 1 Red is different from Object 2 Red), replacement allowed. Permutation with replacement logic.
    * **Solution:** $n=3$ colors, $k=4$ objects. $3^4 = 81$.

83. **Problem:** How many ways can you arrange 3 letters A, B, C?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(3, 3) = 3! = 6$

84. **Problem (Complex):** A bag contains 5 red and 3 blue marbles. If 3 marbles are drawn without replacement, what is the probability that exactly 2 are red?
    * *Identify:* Probability using combinations without replacement.
    * **Solution:**
        * Ways to choose 2 Red from 5: $C(5, 2) = 10$.
        * Ways to choose 1 Blue from 3: $C(3, 1) = 3$.
        * Favorable outcomes (2R, 1B): $C(5, 2) \times C(3, 1) = 10 \times 3 = 30$.
        * Total ways to choose any 3 from 8: $C(8, 3) = \frac{8!}{3!5!} = 56$.
        * Probability = $\frac{30}{56} = \frac{15}{28}$.

85. **Problem:** How many three-letter codes can be formed using A, B, C, D, E if repetition is allowed?
    * *Identify:* Order matters, replacement allowed. Permutation with replacement.
    * **Solution:** $n=5, k=3$. $5^3 = 125$.

86. **Problem:** How many ways can you choose 6 pastries from a bakery that offers 4 different types?
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.
    * **Solution:** $n=4$ types, $k=6$ pastries. $C(4+6-1, 6) = C(9, 6) = C(9, 3) = 84$.

87. **Problem:** How many ways can you arrange the letters in "ARRANGE"?
    * *Identify:* Order matters, non-distinct items (A, R repeat). Permutation with non-distinct items.
    * **Solution:** $n=7$, $n_A=2, n_R=2$. $\frac{7!}{2! 2!} = \frac{5040}{4} = 1260$.

88. **Problem:** How many ways can you choose 2 side dishes from a menu of 8 options, where you are allowed to pick the same side dish twice?
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.
    * **Solution:** $n=8$ options, $k=2$ choices. $C(8+2-1, 2) = C(9, 2) = 36$.

89. **Problem:** How many distinct seating arrangements are possible for 7 guests at a head table?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(7, 7) = 7! = 5040$.

90. **Problem:** How many ways to choose 2 items from set A={a,b} and 3 items from set B={x,y,z,w}? Assume order doesn't matter within the final group of 5.
    * *Identify:* Two independent combinations without replacement. Multiplication principle.
    * **Solution:** Choose from A: $C(2, 2) = 1$. Choose from B: $C(4, 3) = 4$. Total ways = $1 \times 4 = 4$.

91. **Problem:** How many ways can you distribute 5 identical prizes among 10 students (a student can receive more than one prize)?
    * *Identify:* Order doesn't matter (identical prizes), replacement allowed (assigning prize 1 to student A is choosing A; assigning prize 2 to A is choosing A again). Combination with replacement.
    * **Solution:** $n=10$ students, $k=5$ prizes. $C(10+5-1, 5) = C(14, 5) = 2002$.

92. **Problem:** How many different sums of money can be formed using at least one coin from a collection containing a penny, a nickel, a dime, and a quarter?
    * *Identify:* Each coin is either chosen or not. Subset approach. Exclude the empty set.
    * **Solution:** 4 coins. Each has 2 options (in or out). Total possibilities = $2^4 = 16$. Exclude the case where no coins are chosen (1 way). Number of sums = $16 - 1 = 15$.

93. **Problem:** In a game, you roll a standard 6-sided die 3 times. How many different sequences of outcomes are possible?
    * *Identify:* Order matters (sequence), replacement allowed. Permutation with replacement.
    * **Solution:** $n=6, k=3$. $6^3 = 216$.

94. **Problem:** How many ways can you select 12 bottles of soda from 5 different brands available?
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.
    * **Solution:** $n=5$ brands, $k=12$ bottles. $C(5+12-1, 12) = C(16, 12) = C(16, 4) = 1820$.

95. **Problem:** Assigning 3 different tasks to 3 people out of 5 available people.
    * *Identify:* Order matters (Task 1 to Person A != Task 1 to Person B), no replacement (person assigned one task isn't available for another simultaneously, if tasks distinct). Permutation without replacement.
    * **Solution:** $P(5, 3) = \frac{5!}{(5-3)!} = \frac{5!}{2!} = 60$.

96. **Problem:** How many three-letter codes can be formed using A, B, C, D, E if repetition is not allowed?
    * *Identify:* Order matters, no replacement. Permutation without replacement.
    * **Solution:** $P(5, 3) = \frac{5!}{(5-3)!} = \frac{5!}{2!} = 60$.

97. **Problem:** Arrange 2 identical copies of book A, 3 identical copies of book B, and 1 copy of book C on a shelf.
    * *Identify:* Order matters, non-distinct items. Permutation with non-distinct items.
    * **Solution:** $n=6$, $n_A=2, n_B=3$. $\frac{6!}{2! 3! 1!} = \frac{720}{2 \times 6 \times 1} = 60$.

98. **Problem (Complex):** How many ways are there to assign 4 different tasks to 3 employees, if each employee can be assigned multiple tasks and each task must be assigned?
    * *Identify:* Functions from tasks to employees. Each task has 3 choices. Order matters (distinct tasks). Replacement allowed (employee can do multiple tasks).
    * **Solution:** Task 1 has 3 choices (employee). Task 2 has 3 choices... Task 4 has 3 choices. Total ways = $3 \times 3 \times 3 \times 3 = 3^4 = 81$.

99. **Problem:** How many subsets of size 3 can be formed from the set {1, 2, 3, 4, 5, 6}?
    * *Identify:* Order doesn't matter (subset), no replacement. Combination without replacement.
    * **Solution:** $C(6, 3) = \frac{6!}{3!(6-3)!} = \frac{6!}{3!3!} = 20$.

100. **Problem:** How many ways can a band of 4 members arrange themselves on stage?<br>
    * *Identify:* Order matters, no replacement. Permutation without replacement.<br>
    * **Solution:** $P(4, 4) = 4! = 24$.

101. **Problem:** How many ways to form a subcommittee of 6 members from a committee of 11?<br>
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.<br>
    * **Solution:** $C(11, 6) = C(11, 5) = \frac{11!}{6!(11-6)!} = \frac{11!}{6!5!} = 462$.

102. **Problem:** How many 6-character passwords can be formed using letters (a-z) and digits (0-9) if repetition is allowed? (Assume case-insensitive, 36 characters).<br>
    * *Identify:* Order matters, replacement allowed. Permutation with replacement.<br>
    * **Solution:** $n=36, k=6$. $36^6 = 2,176,782,336$.

103. **Problem:** Picking 8 pieces of fruit from a large supply of apples, bananas, and oranges.<br>
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.<br>
    * **Solution:** $n=3$ fruit types, $k=8$ pieces. $C(3+8-1, 8) = C(10, 8) = C(10, 2) = 45$.

104. **Problem:** How many 5-digit numbers can be formed using digits 1, 2, 3, 4, 5 if the number must be even and digits cannot be repeated?<br>
    * *Identify:* Permutation without replacement with constraint. Fix the last digit first.<br>
    * **Solution:** Last digit must be 2 or 4 (2 choices). Arrange the remaining 4 digits in the first 4 positions: $P(4, 4) = 4! = 24$. Total ways = $2 \times P(4, 4) = 2 \times 24 = 48$.

105. **Problem (Complex):** How many ways can 4 Math books and 3 Physics books be arranged on a shelf if no two Physics books can be adjacent? (Assume books of same subject are distinct).<br>
    * *Identify:* Arrangement with separation constraint. Place the larger group first, then slot the smaller group. Permutations.<br>
    * **Solution:**
        * Arrange the 4 Math books: $4! = 24$ ways.
        * This creates 5 slots (\_M\_M\_M\_M\_) for Physics books.
        * Choose 3 of these 5 slots: $C(5, 3) = 10$ ways.
        * Arrange the 3 Physics books in the chosen slots: $3! = 6$ ways.
        * Total ways = (Arrange Math) \* (Choose Slots) \* (Arrange Physics) = $24 \times 10 \times 6 = 1440$.

106. **Problem:** A telephone number in a certain area consists of 7 digits. How many phone numbers are possible if the first digit cannot be 0 or 1?<br>
    * *Identify:* Order matters, replacement allowed, constraint on first digit. Multiplication principle.<br>
    * **Solution:** 1st digit: 8 choices (2-9). Digits 2-7: 10 choices each. Total = $8 \times 10^6 = 8,000,000$.

107. **Problem:** You want to buy 3 donuts from a shop that sells 5 types. How many different selections can you make?<br>
    * *Identify:* Order doesn't matter, replacement allowed. Combination with replacement.<br>
    * **Solution:** $n=5$ types, $k=3$ donuts. $C(5+3-1, 3) = C(7, 3) = 35$.

108. **Problem:** Selecting a team of 11 players from 15 potential players.<br>
    * *Identify:* Order doesn't matter, no replacement. Combination without replacement.<br>
    * **Solution:** $C(15, 11) = C(15, 4) = 1365$.

109. **Problem:** How many ways are there to choose a password consisting of 4 distinct letters followed by 2 distinct digits (0-9)?<br>
    * *Identify:* Independent permutations without replacement. Multiplication principle.<br>
    * **Solution:** Letters: $P(26, 4) = 26 \times 25 \times 24 \times 23 = 358,800$. Digits: $P(10, 2) = 10 \times 9 = 90$. Total = $358,800 \times 90 = 32,292,000$.

110. **Problem:** How many ways to choose 4 chocolates from a box containing 6 identical dark chocolates and 4 identical milk chocolates? (Select 4 chocolates in total).<br>
    * *Identify:* Choosing combinations of non-distinct items. Enumerate compositions.<br>
    * **Solution:** Possible (Dark, Milk) compositions summing to 4: (4,0), (3,1), (2,2), (1,3), (0,4). All are possible since we have enough of each type. Total ways = 5.

---
