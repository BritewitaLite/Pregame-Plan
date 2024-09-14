# Pregame-Plan
Designing an AI play card maker through video analysis of game film.

Prime Gameplan
 
      The concept behind this idea is to save football coaches time when making scout cards. Scout cards can take hours upon hours to design. 
      Coaches have to watch multiple games of an opponent and draw up every play, either by paper and pencil or using a software program where 
      they still have to create the offensive play digitally by hand.
 
      This software program will eliminate all the designing by hand. This concept will take an image of the screen (from footage that is uploaded
      into the software) and automatically draw the play on a scout card. It will show the offensive formation, motions, pass routes, blocking 
      scheme and running backs designed path.  Every scout card will be digitally designed within seconds.
 
      A defensive formation can be selected if you do not have film of the offense playing against the defense you run. The program can also work 
      though to read an opponent's defense and see tendencies of blitzing linebackers and coverages they like to run vs. certain offensive sets.
	
	    The program can be run in live time as the plays are seen by the program so the individual can see the play as well. This way they can make 
      any adjustments to the card they want if they believe there is another way the opponent may run the play. The individual can also number 
      certain players to see tendencies of the offense.  

	    ???? Color coding  the wide receiver passing routes, ball carrier and linemen blocks will enable the players to see exactly what their 
      assignment is when they see the card.


Strategy for Development
  
  First step is to train a machine learning algorithmusing Neural network to recognize the players on the field
  - All players should be recognized as separate entities
      - This needs to only capture the players on hte field, will have to set boundaries by first teaching the AI what the sidelines are and limiting
        the labels to only be within the sidelines
      - Also need to label the refs to know to block them out of the play when transfered to pdf drawing.
    
  - The next step is separating the defense from the offense
        - For simplicity sake the user could tell the program which color jersey the team on offense they are drawing the cards on is wearing (this can
          be later improved upon to train the AI to recognize it themselves)
    
  - We then need to train the AI to recognize each player as a specific position (OL, DL, LB, QB, RB, WR, TE, S, CB) This can go more in depth if we want
        - How do we train this to recognize position (Current thought is it can find the middle entity and count out away from it signaling different positions)???

  After we can correctly have the AI detect each player and give them a proper label, we then will implement video tracking
  - the Video will be broken down into frames and ensure the AI is still capable of tracking each entity as they are blocking eachother and might become hard to distinguish

  Then the AI will draw into a apdf file that has a predrawn player layout for defense depending on what type of defense the user says their own team runs and then draw 
  in the offensive side and the play (Just the lines and circles of each player)
  - The first focus will be drawing in the circles as players, then analyzing the players movements in the video to draw lines for the path they took
      - The first plers whose paths need programmed are the WR, RB, QB, TE
      - Next the Lineman will need done, this will be harder as it is more convoluded with them in a small space
          -  The reason this is last as well is being the blocking will have to be trained to the users imput on what kind of defense they run
          -  If the film is against a 3-4 but they run a 4-3 then the AI needs to change the blocking scheme for how they would block against that

