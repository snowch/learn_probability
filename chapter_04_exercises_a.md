---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
downloads:
  - file: notebooks/chapter_04_exercises_a.ipynb
---

# Exercises for Chapter 4 (Part A)

Here are 30 questions designed to assess a student's ability to differentiate between joint probability, $P(A \cap B)$, and conditional probability, $P(A | B)$, based on contextual information.

---

1.  **University Demographics:** At a certain university, 30% of students major in business. It's also noted that 10% of the total student population are female business majors.
    * Let $B$ be the event that a student is a business major, and $F$ be the event that a student is female.
    * The statement "10% of the total student population are female business majors" translates to which probability expression: $P(B \cap F)$ or $P(F | B)$? Justify your choice.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(B \cap F) = 0.10$. The phrase "10% of the total student population are female business majors" indicates the intersection of the two events – students who belong to *both* categories out of the entire population.
    ```

2.  **Retail Discounts:** A retail outlet notes that 45% of its sales are for electronic items. Among all sales, 15% consist of discounted electronics.
    * Let $E$ be the event that a sale is for electronics, and $D$ be the event that a sale is discounted.
    * The figure "15% of all sales consist of discounted electronics" relates to which probability: $P(E \cap D)$ or $P(D|E)$? Provide the notation and the value.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(E \cap D)$, specifically $P(\text{Electronics} \cap \text{Discounted}) = 0.15$. The phrasing "15% of all sales consist of discounted electronics" implies these sales meet both criteria simultaneously out of all possible sales.
    ```

3.  **Customer Loyalty:** A recent survey indicated that 60% of customers were satisfied with a particular service. Among the group of satisfied customers, 90% stated they would recommend the company.
    * Let $S$ be the event that a customer was satisfied, and $R$ be the event that a customer would recommend the company.
    * The information "Among the group of satisfied customers, 90% stated they would recommend the company" is an example of what type of probability? Express this as $P(A \cap B)$ or $P(A | B)$ using the defined events.

    ```{admonition} Answer
    :class: dropdown
    This represents a conditional probability, $P(R | S) = 0.90$. The phrase "Among the group of satisfied customers" restricts the sample space to only those who were satisfied, which is the hallmark of conditional probability.
    ```

4.  **Regional Climate:** For a specific region, the probability of rain on any day is 0.70. Given that it is raining, the probability of experiencing high humidity is 0.80.
    * Define events for Rain (R) and High Humidity (H).
    * Express the information "Given that it is raining, the probability of experiencing high humidity is 0.80" using your defined events and the correct probability notation.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(H | R) = 0.80$. The phrase "Given that it is raining" explicitly states a condition for the probability of high humidity.
    ```

5.  **E-Learning Success:** A study on online education found that 40% of students choose to enroll in a particular online course. Of the students who enroll, 60% go on to complete the course successfully.
    * Let $E$ be the event a student enrolls and $C$ be the event a student completes the course.
    * The statement "Of the students who enroll, 60% go on to complete the course successfully" describes what kind of probability? Write it using the events $E$ and $C$.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(C | E) = 0.60$. The phrase "Of the students who enroll" indicates that the 60% is conditioned on enrollment.
    ```

6.  **Household Assets:** In a certain town, 70% of households own a car. It is also known that 35% of all households in this town own both a car and have a garage.
    * Consider the events $C$: a household owns a car, and $G$: a household has a garage.
    * Translate the statement "35% of all households in this town own both a car and have a garage" into probability notation.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(C \cap G) = 0.35$. The phrasing "own both a car and have a garage" points to the intersection of the two events relative to all households.
    ```

7.  **Electoral Demographics:** Data from a recent election shows that 55% of the eligible population cast a vote. Furthermore, 30% of the total eligible population both voted and were aged over 65.
    * Let $V$ represent the event that a person voted and $O$ represent the event that a person is over 65.
    * The information "30% of the total eligible population both voted and were aged over 65" is an example of $P(V \cap O)$ or $P(O|V)$? Specify the correct notation and its value.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(V \cap O) = 0.30$. The wording "both voted and were aged over 65" out of the "total eligible population" indicates an intersection.
    ```

8.  **Candidate Screening:** A company is reviewing job applications. They find that 80% of applicants possess a college degree. Among those applicants who have a college degree, 70% also have relevant previous work experience.
    * Let $D$ be the event an applicant has a college degree, and $W$ be the event an applicant has previous work experience.
    * Which probability does the statement "Among those applicants who have a college degree, 70% also have relevant previous work experience" describe: a joint probability or a conditional probability? Provide the specific notation and value.

    ```{admonition} Answer
    :class: dropdown
    This describes a conditional probability, specifically $P(W | D) = 0.70$. The phrase "Among those applicants who have a college degree" sets a condition for the 70%.
    ```

9.  **Dining Habits:** At a bustling restaurant, 90% of diners select a main course. For diners who order a main course, there's a 0.50 probability that they will also order an appetizer.
    * Identify suitable events M (Main Course) and A (Appetizer).
    * Express the information "For diners who order a main course, there's a 0.50 probability that they will also order an appetizer" using probability notation.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(A | M) = 0.50$. The condition "For diners who order a main course" clearly indicates a conditional probability.
    ```

10. **Tech Ownership:** Surveys show that 60% of individuals own a smartphone. It is also found that 25% of all individuals own both a smartphone and a tablet.
    * Let $S$ be the event of owning a smartphone and $T$ be the event of owning a tablet.
    * The statistic "25% of all individuals own both a smartphone and a tablet" corresponds to which of the following: $P(S \cap T)$, $P(S|T)$, or $P(T|S)$? Explain your choice.

    ```{admonition} Answer
    :class: dropdown
    This corresponds to $P(S \cap T) = 0.25$. The phrase "25% of all individuals own both a smartphone and a tablet" means these individuals possess both devices out of the entire population, indicating an intersection.
    ```

11. **Software Adoption:** Data indicates that 75% of users have installed the latest software update. Within the group of users who have updated, 85% report satisfaction with the new version.
    * Let $U$ be the event a user updated and $S$ be the event a user reported satisfaction.
    * The value 85% refers to $P(S \cap U)$ or $P(S|U)$? Write the full expression.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(S | U) = 0.85$. The phrase "Within the group of users who have updated" establishes a condition.
    ```

12. **Horticulture:** A plant nursery stocks various plants. Twenty percent of its inventory consists of perennial flowers. Out of all plants in the nursery, 8% are perennial flowers that are also deer-resistant.
    * Define events $P$: plant is a perennial flower, and $D$: plant is deer-resistant.
    * Which piece of information allows you to write a joint probability? State this probability using your defined events.

    ```{admonition} Answer
    :class: dropdown
    The statement "8% of all its plants are perennial flowers that are also deer-resistant" allows us to write the joint probability $P(P \cap D) = 0.08$. The key is "also" or "and" applied to the entire stock.
    ```

13. **Commuting Choices:** A survey on commuting habits finds that 40% of commuters utilize public transport. If a commuter is known to use public transport, there is a 60% likelihood they also own a personal vehicle.
    * Let $PT$ be the event a commuter uses public transport and $PV$ be the event a commuter owns a personal vehicle.
    * How should the 60% likelihood be expressed in probability notation?

    ```{admonition} Answer
    :class: dropdown
    This should be expressed as $P(PV | PT) = 0.60$. The condition "If a commuter is known to use public transport" signals a conditional probability.
    ```

14. **Library Engagement:** Statistics show 50% of library cardholders visit the library at least monthly. Among all cardholders, 20% are monthly visitors who primarily borrow fiction books.
    * Consider $M$: cardholder visits monthly, and $F$: cardholder borrows fiction.
    * Identify which percentage represents $P(M \cap F)$ and provide its value.

    ```{admonition} Answer
    :class: dropdown
    The statement "20% of all library cardholders are monthly visitors who primarily borrow fiction books" represents $P(M \cap F) = 0.20$. The phrasing indicates both conditions are met by this 20% of the total cardholder population.
    ```

15. **Household Energy:** In a typical household, appliances account for 80% of electricity consumption. Considering only the electricity used by appliances, 30% of that amount is consumed by the refrigerator.
    * Let $A$ be the event that electricity is used by an appliance, and $R$ be the event that electricity is used by the refrigerator.
    * The 30% figure refers to $P(R \cap A)$ or $P(R|A)$? Explain your reasoning.

    ```{admonition} Answer
    :class: dropdown
    This represents $P(R | A) = 0.30$. The phrase "Considering only the electricity used by appliances" restricts the context to appliance usage, hence it's a conditional probability.
    ```

16. **Digital Access:** It's reported that 90% of homes have an internet connection. Furthermore, 70% of all homes possess both internet access and a fiber optic connection.
    * Let $I$ be having internet access and $F$ be having a fiber optic connection.
    * Translate "70% of all homes possess both internet access and a fiber optic connection" into a probability statement.

    ```{admonition} Answer
    :class: dropdown
    This translates to $P(I \cap F) = 0.70$. The term "both...and" applied to "all homes" indicates an intersection.
    ```

17. **Extracurricular Activities:** At a high school, 25% of students are members of the debate club. For those students who are in the debate club, 40% also participate in the drama club.
    * Let $Debate$ be the event a student is in the debate club, and $Drama$ be the event a student is in the drama club.
    * Which probability notation accurately describes the statement "For those students who are in the debate club, 40% also participate in the drama club"?

    ```{admonition} Answer
    :class: dropdown
    This is $P(\text{Drama} | \text{Debate}) = 0.40$. The condition "For those students who are in the debate club" is key.
    ```

18. **Investment Portfolio:** An investor's portfolio consists of 60% stocks. Twenty percent of the investor's total portfolio value is comprised of stocks from international markets.
    * Let $S$ be the event an investment is a stock, and $I$ be the event an investment is in an international market.
    * The value "20% of the investor's total portfolio value is comprised of stocks from international markets" represents which probability: $P(S \cap I)$ or $P(I|S)$?

    ```{admonition} Answer
    :class: dropdown
    This represents $P(S \cap I) = 0.20$. The phrasing "20% of the investor's total portfolio... stocks from international markets" (implying stocks AND international) refers to a portion of the entire portfolio satisfying both conditions.
    ```

19. **Public Health:** In a specific community, 70% of the adult population has received a flu vaccine. Of the vaccinated adults, 95% did not contract the flu during the subsequent season.
    * Define appropriate events for being vaccinated (V) and contracting the flu (F).
    * How would you express the information "Of the vaccinated adults, 95% did not contract the flu" using probability notation? (Hint: consider the event "not contracting the flu").

    ```{admonition} Answer
    :class: dropdown
    Let $F^c$ be the event of not contracting the flu. The information represents $P(F^c | V) = 0.95$. The phrase "Of the vaccinated adults" sets the condition.
    ```

20. **Product Reliability:** A manufacturer observes that 5% of their electronic devices experience a failure within the first year of use. Data also shows that 2% of all devices sold fail in the first year and necessitate a complete replacement.
    * Let $F1$ be the event a device fails in the first year, and $R$ be the event it requires full replacement.
    * Translate the statement "2% of all devices sold fail in the first year and necessitate a complete replacement" into probability notation.

    ```{admonition} Answer
    :class: dropdown
    This is $P(F1 \cap R) = 0.02$. The phrasing "...fail in the first year and require a full replacement" applied to "all devices" signifies an intersection.
    ```

21. **Home Conveniences:** In a survey, 85% of households reported owning a washing machine. Among these households (those with a washing machine), 70% also own a clothes dryer.
    * Let $W$ be owning a washing machine and $D$ be owning a dryer.
    * The 70% figure is an instance of which type of probability? Write the specific probability statement.

    ```{admonition} Answer
    :class: dropdown
    This is a conditional probability: $P(D | W) = 0.70$. The context "Among these households (those with a washing machine)" indicates the condition.
    ```

22. **Reading Habits:** Forty percent of adults subscribe to at least one magazine. Out of all adults, 15% subscribe to a magazine and report reading it thoroughly from cover to cover.
    * Let $S$ be subscribing to a magazine, and $R$ be reading it cover-to-cover.
    * What does "15% of all adults subscribe to a magazine and report reading it thoroughly" represent in terms of probability notation?

    ```{admonition} Answer
    :class: dropdown
    This represents $P(S \cap R) = 0.15$. The phrasing "...subscribe...and read..." out of "all adults" indicates an intersection.
    ```

23. **Urban Transit:** In a metropolitan area, 60% of daily commutes are made using public transportation. If a given commute is by public transportation, there's a 20% chance that it involves at least one transfer.
    * Define events $PT$: commute by public transport, and $T$: commute involves a transfer.
    * Express the 20% chance using these events and the correct probability notation.

    ```{admonition} Answer
    :class: dropdown
    This is $P(T | PT) = 0.20$. The condition "If a given commute is by public transportation" is explicitly stated.
    ```

24. **Mobile Technology:** Current smartphones are advanced: 90% of them include a camera. Looking at the entire market, 80% of all smartphones sold have both a camera and facial recognition capabilities.
    * Let $C$ denote having a camera and $FR$ denote having facial recognition.
    * Which piece of information represents $P(C \cap FR)$? State its value.

    ```{admonition} Answer
    :class: dropdown
    The information "80% of all smartphones sold have both a camera and facial recognition capabilities" represents $P(C \cap FR) = 0.80$.
    ```

25. **Event Logistics:** At a major professional conference, 50% of attendees traveled from out-of-state. For those attendees who came from out-of-state, 70% chose to stay in one of the officially recommended conference hotels.
    * Let $OOS$ be the event an attendee is from out-of-state, and $H$ be the event an attendee stayed in a recommended hotel.
    * The statement "For those attendees who came from out-of-state, 70% chose to stay..." is an example of what? Provide the probability notation.

    ```{admonition} Answer
    :class: dropdown
    This is an example of conditional probability, $P(H | OOS) = 0.70$. The phrase "For those attendees who came from out-of-state" clearly defines the condition.
    ```

26. **Website Analytics:** A popular content website observes that 70% of its daily visitors engage with video content. Across all visitors, 30% both view video content and are subscribed to the site's newsletter.
    * Let $V$ be viewing video content and $N$ be subscribing to the newsletter.
    * Determine whether "30% across all visitors both view video content and are subscribed" is $P(V \cap N)$ or $P(N|V)$, and provide the value.

    ```{admonition} Answer
    :class: dropdown
    This is $P(V \cap N) = 0.30$. "Across all visitors" and "both...and" indicate an intersection of events.
    ```

27. **Digital Banking:** A survey on banking preferences found that 65% of consumers utilize online banking services. Of this group who use online banking, 40% also frequently use a mobile banking application.
    * Consider $OB$: uses online banking, and $MB$: uses a mobile banking app.
    * Interpret "Of this group who use online banking, 40% also frequently use a mobile banking application" as a probability statement.

    ```{admonition} Answer
    :class: dropdown
    This is $P(MB | OB) = 0.40$. The condition is "Of this group who use online banking."
    ```

28. **Culinary Preferences:** A food survey reveals that 70% of respondents enjoy chocolate. Among all respondents, 25% both enjoy chocolate and specifically prefer dark chocolate.
    * Let $C$ be liking chocolate and $D$ be preferring dark chocolate.
    * Translate the statement "25% of all respondents both enjoy chocolate and specifically prefer dark chocolate" into the language of probability.

    ```{admonition} Answer
    :class: dropdown
    This translates to $P(C \cap D) = 0.25$. "Both...and" applied to "all respondents" indicates an intersection.
    ```

29. **Academic Progression:** Statistics show that 80% of high school graduates pursue some form of higher education. If a graduate decides to pursue higher education, there is a 60% probability they will enroll in a four-year university program.
    * Let $HE$ be pursuing higher education and $U$ be enrolling in a university.
    * The 60% probability refers to $P(U \cap HE)$ or $P(U|HE)$? Provide the expression.

    ```{admonition} Answer
    :class: dropdown
    This refers to $P(U | HE) = 0.60$. The condition "If a graduate decides to pursue higher education" is key.
    ```

30. **Online User Behavior:** On a specific e-commerce website, 95% of users visit the homepage during their session. Overall, 50% of all users to the site visit the homepage and also click on a featured promotional banner.
    * Define $H$: user visits homepage, and $B$: user clicks promotional banner.
    * The figure "50% of all users to the site visit the homepage and also click on a featured promotional banner" is an example of which probability type (joint or conditional)? Write the notation.

    ```{admonition} Answer
    :class: dropdown
    This is an example of a joint probability, $P(H \cap B) = 0.50$. The phrasing "visit the homepage and also click" applied to "all users" indicates an intersection.
    ```