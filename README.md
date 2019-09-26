# ActiveTeachingLab
Crafting a Hopfield network as a cognitive model for AurelienNioche/ActiveTeachingModel

## Functionality

### Weights

The network calculates target weights according to the patterns to be recovered.
These are calculated according to the beautiful yellow website.

Adding some or all (default) of these arrays make the actual target weights.
These are used for the network to update its weights until reaching whose values.

Originally, the network weights are initialized at random (-1,0,1).
To do so, the network needs multiple updates of its own weights.
Every update is stored in a weights history array.

#### Learning



#### Forgetting
