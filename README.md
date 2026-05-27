# Wumpus Hunter
Automatically plays and beats the classic terminal game, [Hunt the Wumpus](https://en.wikipedia.org/wiki/Hunt_the_Wumpus), by Gregory Yob in 1973.

<img width="891" height="1200" alt="image" src="https://github.com/user-attachments/assets/7f1feae0-d58f-4c1c-b164-63e52a64d1a1" />

Uncle Bob did an [experiment](https://x.com/unclebobmartin/status/2059604009376260226) to compare human vs AI implementation time. Instead of doing what I was supposed to do, I did the exact opposite. It would be too easy for an AI to implement the game logic itself, since the code exists many times over in its training set, so instead I asked Claude and Codex to collaborate on a solver that would play the game optimally.

You can run it with `hunter.py`. I have been using Eric S. Raymond's implementation of the game, which can be found [here](https://gitlab.com/esr/wumpus).
