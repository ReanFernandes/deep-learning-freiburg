# Please give us feedback to help us improve the exercises. You get 1 bonus point for submitting feedback.

## Major Problems?

The plot attention function with the custom made error that said that pytorch keeps track of gradients threw me off for a very long time. I tried using detach().clone() and the reverse of it but it never worked. After disabling the try/except block i finally saw that my error was that one variable was in cpu while the other one was in cuda, which i easily fixed.

Perhap that way of showing a runtime error could be skipped next time so that we do not get a misleading error and spend a lot of time trying fixes for something that isnt the cause of the issue.

Apart from that, the exercise was helpful, but very heavy.

## Helpful?

Extremely helpful assignment, it was rich in concepts and was the perfect stepping stone for me to really get to know transformers and also gave me so many doubts, which after solving increased my grasp of this concept. thank you so much for this assignment!

## Duration (hours)?

_Please make a list where every student in your group puts in the hours they used to do the complete exercise_
_Example: [5.5, 4, 7], if one of you took 5 and a half hours, one person 4 and the other 7. The order does not matter._
_This feedback will help us analyze which exercise sheets are too time-intensive._

[15,_,_]

## Other feedback?

Not feedback as such, but I didnt quite check whether you had made edits for gpu to be used instead of cpu, so i made them in some places in the code. I know that calls for a deduction in marks, and I just wanted to let you know that next time I will be careful. This time i just didnt have enough time to make the edits before merging to master. 


