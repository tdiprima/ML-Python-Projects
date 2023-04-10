"""
Close pygame window
A video thing has to be running first.
This worked before.  Crap.  TODO: Read.
https://stackoverflow.com/questions/19882415/closing-pygame-window
"""
import pygame
# import time

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.display.quit()
pygame.quit()
exit()
