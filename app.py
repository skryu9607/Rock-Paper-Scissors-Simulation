import pygame
import os
import random
import math
from PIL import Image
WIDTH = 300 * 2
HEIGHT = 300 * 2

BG_COLOR = (255, 255, 255)

ROCK_IMG = os.path.join('assets', 'Rock.png')
PAPER_IMG = os.path.join('assets', 'Paper.png')
SCISSORS_IMG = os.path.join('assets', 'Scissors.png')
def load_image(filename):
	img = Image.open(filename).convert("RGBA")
	return pygame.image.fromstring(img.tobytes(), img.size, img.mode)



IMG_WIDTH = 40
IMG_HEIGHT = 40
STEP_SPEED = 20.0


rock_count = 2
scissors_count = 5
paper_count = 5

# make collide function for irregular objects

def collide(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    
    dist = math.hypot(dx, dy)
    
    if p1.filename == p2.filename:
        pass

    elif dist < IMG_WIDTH:
        tangent = math.atan2(dy, dx)
        angle = 0.5 * math.pi + tangent

        angle1 = 2*tangent - p1.angle
        angle2 = 2*tangent - p2.angle
        speed1 = p2.speed
        speed2 = p1.speed

        (p1.angle, p1.speed) = (angle1, speed1)
        (p2.angle, p2.speed) = (angle2, speed2)

        p1.x += math.sin(angle)
        p1.y -= math.cos(angle)
        p2.x -= math.sin(angle)
        p2.y += math.cos(angle)

        if p1.filename == ROCK_IMG and p2.filename == SCISSORS_IMG:
            setattr(p2, 'filename', ROCK_IMG)
            setattr(p2, 'image', (pygame.transform.scale(load_image(ROCK_IMG), (IMG_WIDTH, IMG_HEIGHT))))

        if p1.filename == SCISSORS_IMG and p2.filename == ROCK_IMG:
            setattr(p1, 'filename', ROCK_IMG)
            setattr(p1, 'image', (pygame.transform.scale(load_image(ROCK_IMG), (IMG_WIDTH, IMG_HEIGHT))))

        if p1.filename == ROCK_IMG and p2.filename == PAPER_IMG:
            setattr(p1, 'filename', PAPER_IMG)
            setattr(p1, 'image', (pygame.transform.scale(load_image(PAPER_IMG), (IMG_WIDTH, IMG_HEIGHT))))

        if p1.filename == PAPER_IMG and p2.filename == ROCK_IMG:
            setattr(p2, 'filename', PAPER_IMG)
            setattr(p2, 'image', (pygame.transform.scale(load_image(PAPER_IMG), (IMG_WIDTH, IMG_HEIGHT))))

        if p1.filename == PAPER_IMG and p2.filename == SCISSORS_IMG:
            setattr(p1, 'filename', SCISSORS_IMG)
            setattr(p1, 'image', (pygame.transform.scale(load_image(SCISSORS_IMG), (IMG_WIDTH, IMG_HEIGHT))))

        if p1.filename == SCISSORS_IMG and p2.filename == PAPER_IMG:
            setattr(p2, 'filename', SCISSORS_IMG)
            setattr(p2, 'image', (pygame.transform.scale(load_image(SCISSORS_IMG), (IMG_WIDTH, IMG_HEIGHT))))

class Object:
    def __init__(self, position, filename):
        self.x, self.y = position
        self.filename = filename
        self.speed = 0
        self.angle = 0
        self.image = pygame.transform.scale(load_image(filename), (IMG_WIDTH, IMG_HEIGHT))
    
    def display(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        self.x += math.sin(self.angle) * self.speed
        self.y -= math.cos(self.angle) * self.speed
    
    def action(self):
        pass
    
    def bounce(self):
        if self.x > WIDTH - IMG_WIDTH:
            self.x = 2 * (WIDTH - IMG_WIDTH) - self.x
            self.angle = - self.angle

        # right wall
        elif self.x < 0:
            self.x = - self.x
            self.angle = - self.angle
        
        # floor
        if self.y > HEIGHT - IMG_HEIGHT:
            self.y = 2 * (HEIGHT - IMG_HEIGHT) - self.y
            self.angle = math.pi - self.angle

        # ceiling
        elif self.y < 0:
            self.y = - self.y
            self.angle = math.pi - self.angle
        
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Rock Paper Scissors')

final_list = []
TARGET_FPS = 30.0
dt = TARGET_FPS / 1200.0



for n in range(rock_count):
    x = random.randint(IMG_WIDTH, (WIDTH - IMG_WIDTH))
    y = random.randint(IMG_HEIGHT, (HEIGHT - IMG_HEIGHT))

    item = Object((x, y), ROCK_IMG)
    item.speed = STEP_SPEED * dt
    item.angle = random.uniform(0, math.pi * 2)

    final_list.append(item)


for n in range(paper_count):
    x = random.randint(IMG_WIDTH, (WIDTH - IMG_WIDTH))
    y = random.randint(IMG_HEIGHT, (HEIGHT - IMG_HEIGHT))

    item = Object((x, y), PAPER_IMG)
    item.speed = STEP_SPEED * dt
    item.angle = random.uniform(0, math.pi * 2)
    final_list.append(item)



for n in range(scissors_count):
    x = random.randint(IMG_WIDTH, (WIDTH - IMG_WIDTH))
    y = random.randint(IMG_HEIGHT, (HEIGHT - IMG_HEIGHT))

    scissors = Object((x, y), SCISSORS_IMG)
    scissors.speed = STEP_SPEED * dt
    scissors.angle = random.uniform(0, math.pi * 2)

    final_list.append(scissors)

running = True

while running:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BG_COLOR)

    for i, item in enumerate(final_list):
        item.move()
        item.bounce()

        for item_two in final_list[i+1:]:
            collide(item, item_two)
        item.display(screen)

    pygame.display.flip()
