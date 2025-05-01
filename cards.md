Problem 1: Probability of Drawing a Specific Card First
* Question: If you shuffle a standard 52-card deck and draw just one card, what is the probability that it is the Ace of Spades (A♠)? * Thinking Process:
   * How many possible outcomes are there? (How many different cards could you draw?)
     * There are 52 cards in the deck, so there are 52 possible outcomes.
   * How many of those outcomes match what we want (the Ace of Spades)?
     * There is only 1 Ace of Spades in the deck. So, there is 1 favorable outcome.
   * Probability = (Favorable Outcomes) / (Total Possible Outcomes)
 * Calculation:
   * Probability (A♠) = 1 / 52
 * Result: The probability of drawing the Ace of Spades first is 1 in 52, or about 1.92%.
  
Problem 2: Probability of Drawing Any Card of a Specific Rank First
 * Question: If you draw just one card from a standard 52-card deck, what is the probability that it is any Ace?
 * Thinking Process:
   * Total possible outcomes are still 52 (any card in the deck).
   * How many outcomes match what we want (any Ace)?
     * There are 4 Aces in the deck (A♥, A♦, A♣, A♠). So, there are 4 favorable outcomes.
   * Probability = (Favorable Outcomes) / (Total Possible Outcomes)
 * Calculation:
   * Probability (Any Ace) = 4 / 52
 * Simplifying: The fraction 4/52 simplifies to 1/13.
 * Result: The probability of drawing any Ace first is 4 in 52 (or 1 in 13), which is about 7.69%.
   
Problem 3: Probability of Getting a Pair in a 2-Card Hand
 * Question: Now, let's draw just two cards from the deck. What is the probability that those two cards form a pair (e.g., two Kings, two 3s)?
 * Thinking Process: This requires combinations.
   * Total Possible Outcomes: How many different ways can you draw 2 cards from 52? Since the order doesn't matter ({K♥, 7♦} is the same as {7♦, K♥}), we use combinations.
     * Total 2-card hands = C(52, 2) = \binom{52}{2} = \frac{52!}{2!(52-2)!} = \frac{52 \times 51}{2 \times 1} = 1326.
     * There are 1,326 possible unique 2-card hands. This is our denominator.
   * Favorable Outcomes (Ways to get a Pair): How many of those 1,326 hands are pairs? Let's figure this out step-by-step:
     * Step 2a: Choose the rank for the pair.
       * Concept: We need to determine what kind of pair we could form. What are the possibilities for the rank? A pair could be Aces, or Kings, or Queens, and so on, down to Twos.
       * Listing the Types:
         * A pair of Aces
         * A pair of Kings
         * A pair of Queens
         * ... (etc.) ...
         * A pair of 2s
       * Counting the Types: How many different categories of pairs are there based on rank? There are exactly 13, one for each rank.
       * Meaning: This step represents deciding which of these 13 categories of pair we are considering. When counting all possible pairs, we need to account for all 13 of these rank possibilities.
       * Result: There are 13 ways to choose the rank for the pair.
     * Step 2b: Choose the 2 specific cards of that rank.
       * Concept: Once we've decided on a rank (say, we decided on 'Sevens' in Step 2a), we need to figure out how many ways we can actually make a pair of Sevens using the cards in the deck.
       * Available Cards: There are 4 cards of each rank (e.g., 7♥, 7♦, 7♣, 7♠).
       * Choosing the Pair: We need to choose exactly 2 of these 4 cards. Since order doesn't matter ({7♥, 7♦} is the same pair as {7♦, 7♥}), we use combinations.
       * Calculation: Ways to choose 2 cards from 4 = C(4, 2) = \binom{4}{2} = \frac{4!}{2!(4-2)!} = \frac{4 \times 3}{2 \times 1} = 6.
       * Meaning: For any rank chosen in Step 2a, there are 6 specific combinations of 2 cards that form a pair of that rank. (e.g., the 6 possible pairs of Sevens are {7♥, 7♦}, {7♥, 7♣}, {7♥, 7♠}, {7♦, 7♣}, {7♦, 7♠}, {7♣, 7♠}).
       * Result: There are 6 ways to choose the 2 specific cards, once the rank is known.
     * Total Ways to Form a Pair: Now we use the Multiplication Principle. For each of the 13 possible ranks (Step 2a), there are 6 ways to form the actual pair (Step 2b).
       * Ways = (Ways to choose rank) \times (Ways to choose the 2 cards)
       * Ways = 13 \times 6 = 78.
       * There are 78 different possible pairs you can draw in a 2-card hand. This is our numerator.
   * Calculate Probability:
     * Probability (Pair in 2 cards) = (Ways to get a Pair) / (Total Possible 2-card hands)
     * Probability = 78 / 1326
 * Simplifying: The fraction simplifies to 1 / 17.
 * Result: The probability of drawing a pair when you draw just two cards is 1 in 17, or about 5.88%.
   
Problem 4: Probability of Getting Four of a Kind (in a 5-card hand)
 * Question: Let's go back to 5-card hands. What is the probability of being dealt Four of a Kind (e.g., four Jacks and a Seven)?
 * Thinking Process: We use the same principles: combinations and the multiplication principle, but now for a 5-card hand.
   * Total Possible Hands: We need the total number of unique 5-card hands.
     * Total Hands = C(52, 5) = 2,598,960. (Denominator)
   * Favorable Outcomes (Ways to get Four of a Kind): Build the hand step-by-step.
     * Step 4a: Choose the rank for the four cards. (Which rank will you have 4 of?)
       * There are 13 ranks. \mathbf{13} ways (C(13, 1)=13).
     * Step 4b: Choose the four specific cards of that rank. (If you chose Jacks, you need J♥, J♦, J♣, J♠).
       * You must choose all 4. There's only \mathbf{1} way (C(4, 4)=1).
     * Step 4c: Choose the fifth card (the "kicker"). This card can be any card not of the rank chosen in Step 4a.
       * There are 52 - 4 = 48 cards remaining in the deck that don't match the rank of the four-of-a-kind. We need to choose 1 of these.
       * There are \mathbf{48} ways (C(48, 1)=48).
     * Total Ways for Four of a Kind: Use the multiplication principle.
       * Ways = (Ways to choose rank) \times (Ways to choose 4 cards) \times (Ways to choose 5th card)
       * Ways = 13 \times 1 \times 48 = 624. (Numerator)
   * Calculate Probability:
     * Probability (Four of a Kind) = (Ways to get Four of a Kind) / (Total Possible Hands)
     * Probability = 624 / 2,598,960
 * Simplifying: This fraction simplifies to 1 / 4165.
 * Result: The probability of getting Four of a Kind is 624 in 2,598,960 (or 1 in 4,165), which is about 0.024%.
We've successfully calculated the probability for a simple pair in a 2-card draw, and for the much rarer Four of a Kind in a 5-card hand. Both used the ideas of choosing ranks and choosing specific cards. The Full House calculation will combine these ideas. Are you ready to tackle the Full House problem now?
