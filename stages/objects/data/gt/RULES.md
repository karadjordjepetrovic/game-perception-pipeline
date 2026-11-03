# Labelling rules

Rules for hand-labelling ground truth in this project. Every clip follows the same rules, so scores stay comparable. When a case isn't covered, decide, then add the rule here.

## Class

One class: `character`. Any human character in the game world, enemy or teammate.

## What to label

- Characters whose box is at least **20 pixels tall**. Smaller ones are skipped.
- Partly visible characters (behind cover, at the screen edge), as long as you can tell it's a character.
- Blurry characters (motion blur, smoke, muzzle flash), if you can still tell it's a character.
- Characters seen through a scope or sight, as they appear on screen.

## What not to label

- **The player's own arms and weapon** (the first-person view model).
- Anything on the **HUD**: minimap icons, kill-feed portraits, scoreboard, menus.
- Name tags, markers or icons floating above characters. They are not part of the box.
- Characters on in-game screens, posters or reflections.
- Dead bodies. A track ends when the character is down after the death animation; mark it **outside** from that frame.
- Killcam, spectator or menu frames. Avoid them when cutting clips.

## Box

- Draw the box around the **visible part** only, not the estimated full body.
- Tight fit: the box touches the outermost visible pixels (head, feet, weapon held by the character).
- When the visible part changes, adjust the box on that frame.

## IDs and tracks

- One track per character, with one ID from the moment it appears until it leaves.
- **Briefly hidden** (behind cover or another character) and clearly the same character when it reappears: keep the same ID, and mark it **outside** while fully hidden.
- **Leaves the screen and comes back** (e.g. during a camera turn): same ID only if you are sure it's the same character; otherwise start a new track.
- Fully hidden or off screen: always marked **outside**, never a guessed box.

## Interpolation in CVAT

- Set a keyframe at least every **10 frames**, and whenever the character changes direction, speed or visible size.
- Scrub through the interpolated frames afterwards and fix any box that drifts off the character.
