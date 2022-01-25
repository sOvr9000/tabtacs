
The accuracy of a hueristic value of a TableTactics position is crucial for most kinds of AI to come up with a good move in that position, namely the minimax and DQL algorithms.

In DQL, the reward function (observed by the agent) should be equivalent to the heuristic function used by other algorithms.

The following reward/heuristic function is used:

$$H(s)=\ln\left [\frac{(\sum_{j}v_{0,j}h_{0,j})^2}{\sum_{j}(h_{0,j})^2}\right ]-\ln\left [\frac{(\sum_{j}v_{1,j}h_{1,j})^2}{\sum_{j}(h_{1,j})^2}\right ]$$

$$=\ln\left [\left (\frac{\sum_{j}v_{0,j}h_{0,j}}{\sum_{j}v_{1,j}h_{1,j}}\right )^2\cdot\frac{\sum_{j}(h_{1,j})^2}{\sum_{j}(h_{0,j})^2}\right ],$$

where $s$ is the current state, $h_{a,j}$ is the $j^{th}$ piece of army $a$ in state $s$, and $v_{a,j}$ is the handcrafted, estimated value of the $j^{th}$ piece of army $a$ in state $s$.

In English, this calculation considers the following things:

- The sum of the square of the hitpoints of each piece, for each army.
- The square of the weighted sum of the hitpoints of each piece, for each army.

The intuition behind this formula revolves around the idea that an army should be stronger than another army if that army has the same number of total hitpoints but fewer pieces.  Specifically, **the square of a sum grows faster than a sum of squares**.

That encourages an algorithm to search for positions valuing the number of pieces over the hitpoints of each piece.  Both factors are still important, and they do both increase the heuristic, but having more pieces (being more offensive) will result in a larger heuristic value than having the same number of pieces with more hitpoints (being less offensive).

Obviously, that is not a perfect strategy, since it is only a heuristic, but it should be good enough to start with.

The logarithm is in place to keep the heuristic from getting too large.  If a player has a significant advantage, it suffices to say their heuristic value is over some small threshold value like `+5`.

As far as the values of $v_{a,j}$ go, the following definition should work for now:

- Noble: 18
- Fighter: 15
- Thief: 14

This is a weighted sum of their attributes, where `speed` has a weight of $2$ and `hitpoints` has a weight of $3$.
