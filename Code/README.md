# Source layout

`Models/` contains model-specific architectures, preprocessing, profiles and
shared numerical components. `Dataset/` contains image preparation utilities.
The public repository-root runner handles new user datasets and direct testing.

Historical model modules retain their original protocol contracts for the study
and should not be used as a substitute for the generic data interface. Model
parameter profiles distinguish publication evidence from project adaptations.
