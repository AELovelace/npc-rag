"""Additional independent intent pairs; punctuation/spelling variants are NOT extra rows.

Each line is GAME | CHAT. Groups are authored families kept within one partition.
These are routing examples, not assertions that a queried mechanic exists.
"""

PAIRS = {
    "short-controls": """
inventory shortcut|hehehe
hide minimap|phew
chat font size|yikes
rotate without walking|woah
mute area chat|uh huh
exit dialogue box|huzzah
toggle music|welp
move chat window|oof ouch
controller bindings|neat stuff
zoom reset|😅
unfocus text box|🌻
scroll battle log|✨🌙
fullscreen toggle|🎉
touchscreen movement|🤔
unstick movement keys|zzzz
reopen tutorial card|erm right
show damage numbers|eep
hide action bar|aw shucks
keyboard remapping|teehee
open settings panel|huh interesting
""",
    "short-needs": """
protection absorbency limit|im giggling
continence recovery rate|feeling sleepy
replace saturated protection|not much happening
toilet interaction key|i missed chatting
pressure gauge meaning|rough day honestly
sleeping hunger drain|oops embarrassing moment
rest station price|that was awkward lol
clothing bulk penalties|im blushing now
laundry service directions|just sighing
empty food bar|feeling a bit silly
wet clothing effects|that loss was sad
bed recovery speed|any comforting words
curse removal cost|a hug sounds nice
capacity overflow rules|what a lovely evening
washroom map marker|laugh with me
bladder gauge colour|dont mind my grumbling
continence stat cap|im just venting
changing room access|lets smile together
eat during battle|i needed that laugh
drink item cooldown|nice weather huh
""",
    "short-world": """
utopia clinic directions|enjoy utopias skyline
hub travel fee|which skyline charms you
weekly dungeon reset|do ruins scare you
arena payout schedule|does cheering make you happy
party beacon range|do you enjoy crowds
story orb purpose|tell a made up legend
patron vow penalties|which patron seems stylish
quest marker colours|gold is a pretty colour
gathering reset time|trees look peaceful
bank storage ceiling|shiny coins make me smile
shop permit price|would you enjoy shopkeeping
crafting station map|does crafting seem relaxing
fishing hotspot reset|do fish amuse you
companion hire duration|do you miss travelling
dungeon entry requirement|what scenery comforts you
defeat recovery timer|are you afraid of defeat
vow progress display|do you value promises
reward chest ownership|what treasures make you happy
town portal unlock|which portal looks prettier
arena round limit|would you watch a tournament
""",
    "short-systems": """
rename character fee|whats your favourite nickname
account link expiry|are you a computer program
roster character limit|who wrote your dialogue
restore lost connection|do you think in words
claim daily reward|can you remember dreams
sell excess armour|what makes you feel real
unequip cursed boots|do you have a birthday
split item stack|can you hear my voice
join invitation expiry|are your replies generated
trade confirmation button|who designed your personality
buy order cancellation|do you ever get bored
merchant restock clock|is your smile animated
recipe ingredient order|can you invent a song
job experience display|do you like being a guide
magic resource refill|whats your happiest thought
dexterity action gauge|are you secretly a poet
front row swapping|do you have imaginary friends
submit combat effect|is your name meaningful
party leader transfer|what should i call you
bank item retrieval|can you imagine colours
""",
    "you-help": """
can you show me the route to a bank terminal|what is the capital of peru
could you explain the difference between submit and run|how many sides has a hexagon
would you help me allocate my fighters starting points|who painted the mona lisa
can you tell me which key opens the items panel|what planet is closest to the sun
could you find the rule for claiming weekly rewards|how many centimetres are in a metre
can you explain why my protection overflows|what does photosynthesis mean
would you list the stations used for refining|which ocean borders chile
could you describe how gathering resets|what is twelve times nine
can you help me remove this curse from my helmet|what year did the first moon landing happen
would you tell me where to change my characters face|what day follows tuesday
can you explain why my shop sale paid less than listed|what language is spoken in brazil
could you show me how to invite another adventurer|why does bread rise
can you explain what my patron expects from me|what is the square root of sixty four
would you tell me how to bank arena winnings|what is the longest river in south america
could you help me move an item out of storage|how many hours make a week
can you explain why my potion button is disabled|why is the sky blue in real life
would you explain the purpose of the story orbs|when does the northern hemisphere have summer
can you tell me whether eating changes my combat stats|who invented the printing press
could you help me find my active quest objectives|what is the chemical symbol for gold
can you explain the maximum number of party members|how do you say good morning in french
""",
    "game-orders": """
show the key for opening my quest journal|i just finished washing dishes
describe the rules for spending skill points|my bus was late today
tell me the price of hiring a companion|the kettle is boiling irl
explain which rooms let me change clothing|i have a dentist appointment tomorrow
give me directions to a town service desk|my cat is sleeping on my chair
list the effects of being dehydrated in combat|i finally finished my homework
walk me through crafting a meal|i need to go cook actual dinner
tell me where the party invitation appears|my real garden is full of weeds
explain when a shop order pays its seller|work was exhausting this afternoon
describe what death does to carried inventory|im waiting for my laundry at home
show me how to select a different combat target|my neighbour brought me cookies
tell me what colour marks the tracked quest|its raining outside my window
explain where merchant inventory comes from|i bought a new real world keyboard
list the controls for interacting with a bed|im eating toast while we talk
give the conditions for opening the weekly dive|my dog wants a walk
explain what a patrol beacon does for a party|my phone battery is almost dead irl
tell me how to reconnect my game account|ive been learning to paint
describe the rules for selling a cursed item|my coffee went cold at my desk
show the way to the nearest cooking fixture|i have a meeting in ten minutes
explain whether the dungeon boss respawns|my shoes got muddy on my commute
""",
    "game-problems": """
inventory panel vanished and i cant reopen it|*offers a friendly handshake*
my character keeps walking after i release the key|*gives you a little bow*
the quest reward button stays grey even though im done|*sits nearby quietly*
im trapped behind a fixture with no walkable tile|*waves a tiny flag*
my shop listing disappeared but i got no payment|*hums a happy melody*
the game disconnects every time i open the roster|*does an awkward curtsy*
my combat gauge stopped filling|*pretends to be a dragon*
i cant accept this invite while the party menu is open|*tips an imaginary hat*
my equipped boots wont come off|*draws a heart in the air*
the text box eats my movement keys|*hands you a pretend flower*
my gathering node is still empty after reset|*leans back and stretches*
the game crashes when i select my second character|*claps enthusiastically*
my bank withdrawal is stuck pending|*spins in a circle*
the portal keeps returning me to the same tile|*makes a silly face*
i bought food but it never appeared in my bag|*nods thoughtfully at you*
the tutorial card wont advance when i click|*gazes at the pretend stars*
my companion timer ended but hiring is still disabled|*puts on an imaginary crown*
the crafting station refuses ingredients i already own|*whistles a little tune*
i cant reach the confirmation button on mobile|*gives an exaggerated salute*
my health stays at zero after the recovery timer|*settles down beside the fire*
""",
    "implicit-needs": """
im out of spell energy halfway through this dungeon|that dungeon was miserable but im fine now
my gear is damaged and the boss room is next|i lost that fight yesterday and just wanted to vent
i have no spare protection and this one is soaked|i had an embarrassing accident earlier lol
my character is thirsty with no drinks left|the last quest annoyed me but its finished
i used my last healing item before the arena|i sold my favourite hat and feel sentimental
my food bar keeps falling while im resting|the tournament loss still stings emotionally
the clothing i equipped is too bulky to fit together|i miss the party we had last week
i need a place to store these extra gathering drops|that boss looked terrifying haha
my bag is full and the quest wants another item|i felt silly when everyone saw me trip
my party has lost its leader mid dungeon|i liked that adventure even though we lost
the only exit i can see is locked|that shopkeeper sounded grumpy lol
i have a curse and no idea where the healer is|i keep replaying that funny moment in my head
my companion vanished while its contract was active|i was embarrassed but i dont need advice
im too weak for this fight and need a safer route|i just want sympathy about yesterday
my quest says track an objective but no marker appears|that scenery took my breath away
the shop fee is more than all my coins|it felt good to finish that story
my protections already full after one accident|i feel proud of beating my nerves
the game says im over capacity and cannot collect loot|i was disappointed with my luck earlier
i need to return to town before hunger empties|im annoyed at that loss just let me grumble
my account link expired before i finished registration|that whole adventure felt like a dream
""",
    "lore-and-advice": """
which patron is associated with this shrine|which patron would you invite to tea
what role does pip have in the towns story|does pip seem like a fun friend to you
what is the history of the weekly dungeon in the lore|would you enjoy living near that dungeon
why do the towns have different patron banners|which banner do you think is prettiest
what does a story orb represent in the setting|do story orbs remind you of stars
where can i read about the founding of utopia|does utopia feel welcoming to you personally
which named npc handles account services|which npc would you choose as a pen pal
who owns the arena in the game world|do you find arenas exciting to watch
what is the purpose of the shrine in this town|would you enjoy sitting near that shrine
which route connects the starter hubs|which road looks more romantic to you
which meal should i craft for stronger attacks|what meal would you enjoy for its taste
what stats should i raise for faster turns|do you admire quick witted characters
which spell would help my party survive|what sort of magic looks beautiful to you
where should a beginner gather useful ingredients|which gathering spot feels most peaceful to you
should i bank my arena winnings before another round|would you enjoy cheering for me in the arena
which protection should i wear for a long dungeon trip|which outfit do you personally find charming
should i finish this story quest before taking the dive|do you enjoy stories about brave adventurers
what equipment would improve my defence|do you like the look of polished armour
which class should i choose for healing allies|which class would fit your personality best
where should i spend my first skill points|what skill would you enjoy having for fun
""",
    "mixed-intents": """
hiya friend tell me how i can recover after a defeat|hi friend tell me a completely fictional joke
thanks for chatting can you explain party beacons|thanks for chatting your company means a lot
im embarrassed but where do i swap saturated protection|im embarrassed can you say something kind
good to see you what does this curse icon mean|good to see you what sort of music cheers you up
that scared me how can i escape the battle|that scared me could we talk about something funny
lovely evening can you show me the crafting recipe list|lovely evening would you like to imagine a picnic
sorry to bother you my character cannot leave the bed|sorry to bother you i just felt lonely
explain the shop tax and thanks for your patience|thanks for being patient with my rambling
how do i claim my patron reward cheers|cheers to our imaginary friendship
im worried will this dungeon reset remove my progress|im worried about tomorrow at work
hello pal which control selects an enemy|hello pal have you got a funny nickname
that was upsetting where can i refill my spell energy|that was upsetting id like a comforting story
thanks again what makes the gold quest route disappear|thanks again you made me feel welcome
im confused could you explain the difference between coins and stars|im confused by how fast this week went
hi guide tell me where to collect an expired shop order|hi guide i just wanted to wish you well
id appreciate it if you explained the account link process|id appreciate a silly poem about friendship
please tell me why my party invite expired thank you|please tell me something nice about your day
wow that hurt which item can heal the damage|wow that hurt my pride yesterday haha
hey there can you help me get this cursed ring off|hey there do you like the sparkle on this ring
explain why my quest is locked please and thank you|please and thank you are my favourite polite words
""",
    "terse-requests": """
teach me refining|tell a pirate joke
map legend help|make up a riddle
explain arena banking|whats six plus seven
quest reward missing|whats todays weekday
party invite bug|who built the pyramids
armour slot locked|whats a lunar eclipse
help crafting bread|why do cats purr
shop proceeds missing|name a european capital
explain battle rows|translate hello into spanish
help with vows|how old is earth
recipe wont unlock|sing a nonsense verse
coin transfer rules|is pluto a planet
patron shrine location|invent a pet name
town return item|whats a prime number
sell stack help|tell me about penguins
food bonus duration|do penguins have knees
account linking broken|why does ice float
character creation stuck|whats the current year
portal travel requirements|explain real world gravity
weekly reward cap|which continent is kenya
""",
    "extra-boundaries": """
can u help refill mp|can u make me smile
pls explain that stat|pls tell a bedtime tale
where r my stored items|where r your manners hehe
how to increase lvl|how was your afternoon
my hp wont recover|my real headache is annoying
need a fresh protection item|need a little encouragement
rn i cant exit combat|rn im listening to jazz
thirst bar flashing red|my real tea tastes awful
which town sells potions|which town do you adore
why is walking disabled|why do you look so pleased
can you point out the services icon|can you think of a silly rhyme
how many stars does renaming require|how many stars would you wish upon
my account code wont paste|my real laptop fan is noisy
cant find the claim reward control|cant believe how quickly today passed
need instructions for switching patrons|need a goofy distraction
where do quest items get stored|where do your daydreams take you
how does the skill refund work|how does friendship feel to you
what happens to gear when my hire expires|what would your dream companion be like
why is the rest station refusing payment|why does your cloak look so soft
how do i stop following a quest route|how do you imagine the perfect afternoon
please explain the job progress gauge|please imagine a talking teapot
can i gather while carrying too many items|can i share a funny story with you
how do i move coins into safe storage|how do you feel about sunny mornings
need to locate the mirror service|need to vent about my commute
do meal effects expire while logged out|do you prefer sweet or salty flavours
can you tell me what interrupted my rest|can you tell me your favourite imaginary animal
what does the empty protection indicator mean|what does the colour blue make you feel
how can my party travel together|how can we invent a fun secret handshake
my active quest disappeared after reconnecting|my real internet has been annoying lately
help me understand the companion upkeep rules|help me write a goofy greeting
""",
}

# Development wording was authored separately, not generated from the training pairs.
DEV = {
    "dev-interface": """
cant close the roster screen|ahh fair enough
need the journal hotkey|mhm gotcha
tap controls not responding|lol thats wild
how turn off game sound|ooh sparkly
lost the quest panel|🙂🌸
menu covering my buttons|haha whoopsie
chat wont let me walk|🫶
where settings on phone|ahem anyway
show old battle messages|hehe thats cute
can u fix stuck movement|nah im good
""",
    "dev-care": """
need clean pants ingame|feeling awkward tonight
absorbency numbers meaning|please cheer me up
can u explain pressure|that was embarrassing yesterday
where wash dirty gear|i just need sympathy
why cant i hold it ingame|my nerves are acting up irl
how refill hunger fast|i ate a real sandwich
bed not restoring stats|my actual bed is comfy
need toilet directions pls|i should take a break irl
bulk limit on clothes|do you enjoy fancy clothes
what changes continence|lets talk about happy memories
""",
    "dev-economy": """
shop licence coins needed|do you like shiny trinkets
cant retrieve bank deposit|whats your dream present
how much to uncurse|im saving for real shoes
sold sword no coins|retail work tired me out
when stalls restock|do you enjoy browsing markets
cancel outstanding buy order|my online parcel arrived today
bag space all used|my desk is a mess irl
can u show shop rules|what makes a good friend
transfer bank item pls|lets pretend were wealthy
why sale payment short|id love a real holiday
""",
    "dev-combat": """
need mana before next fight|that battle music was lovely
move ally to back row|do you admire brave people
what fuels action meter|that old defeat made me sad
can u explain surrender|im not seeking help just venting
arena pot claim button|would you cheer for me
stuck waiting after defeat|yesterdays fight was so frustrating
how pick spell target|your magic looks wonderful
can potions be reused|tell a story about a wizard
attack control greyed out|do you ever dream of flying
need stronger defence build|that monster looked hilarious
""",
    "dev-lore": """
tell me this shrines backstory|which shrine do you find lovely
where does this hub road lead|would you enjoy a walk here
what does pip do here|do you like pips hat
explain the patron traditions|which patron seems friendliest to you
whats the orb lore|invent a myth about moonbeams
where is the towns clinic|do you find this town cosy
can u name the starter hubs|which hub would you paint
why these banners in lore|do you like these banner colours
who gives this story quest|who would you invite for supper
whats this dungeon called|does this dungeon give you chills
""",
    "dev-social-world": """
how invite my teammate pls|what is brazils capital
party member cap pls|whats seventeen minus eight
my invite timed out|whats a leap year
need party beacon instructions|name jupiters largest moon
which fixture refines ore|why does seawater taste salty
how get cooking recipes|whats the date tomorrow
resource node wont refill|who wrote pride and prejudice
how claim patron prize|why do leaves change colour
is hire time running offline|what time zone is london in
companion contract price pls|how far is the moon
""",
    "dev-bugs": """
login link wont copy|are you running on a server
cannot finish character setup|is someone writing your replies
my new hairstyle didnt save|can you recognise emotions
touch map sends me backwards|do you have secret hobbies
quest completed reward unclickable|can you think without speaking
craft button ignores me|are you a simulated person
my party list is blank|what kind of model powers you
stuck loading when entering town|do you have your own opinions
equipment panel refuses to open|whats it like being an npc
character missing from roster|would you like a different name
""",
    "dev-mixed": """
hiya how do i open storage|hiya *offers a warm smile*
thanks wheres the recipe menu|thanks *gives a friendly wink*
im upset can u explain recovery|im upset just sit with me
help me choose mage stats cheers|cheers *raises a pretend cup*
morning need directions to shops|morning *waves from across the room*
pls explain dungeon limits ty|ty *blows an imaginary kiss*
hello why wont this ring unequip|hello *offers a paper flower*
sorry where do i track quests|sorry *shuffles awkwardly*
im scared how do i retreat|im scared tell me a silly tale
nice chat now how do i heal|nice chat *settles onto a cushion*
""",
    "dev-advice": """
recommend gear for my mage|recommend a funny imaginary nickname
which meal buffs defence|which meal sounds tastiest to you
where should i grind gathering|where would you relax for fun
what starter class heals|what class feels like your personality
should i collect arena pot now|would you have enjoyed that contest
best way to use skill points|whats your favourite fictional power
where go after welcome quest|where do you dream of travelling
help me survive this dungeon|help me invent a fairy tale
what protection lasts longer|what style of outfit pleases you
which patron suits magic builds|which patron would make you laugh
""",
    "dev-short": """
mp refill method|just chilling here
clothes cleaning fee|home from work finally
tracked marker absent|my cat says hello
rename currency needed|real life chores await
bank columns navigation|waiting for pizza irl
new quest locked|im sipping cocoa
weekly chest access|its late here irl
resting food penalty|feeling nostalgic today
refining ingredient amounts|im doodling on paper
town waypoint activation|my train just arrived
""",
}
