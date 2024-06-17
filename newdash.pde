import processing.serial.*;
import java.io.File;
import java.util.ArrayList;

Serial myPort;
String inString;
String portName;
float yaw = 0, pitch = 0, roll = 0;
PShape wrench;
PFont coolvetica;
PFont fira; // Fira font for actions panel
float deltaTime = 0.0025; // 400 updates per second

PImage[] images;
String[] imagePaths = {"xyz_plot.png", "xy_plot.png", "xz_plot.png", "yz_plot.png"};
long[] lastModifiedTimes;
boolean isReloadingImages = false; // Flag to indicate image reloading

ArrayList<String> actions = new ArrayList<String>();
boolean isMoving = false;
boolean checkForStop = false;
long stopStartTime;
float threshold = 10;

void setup() {
  fullScreen(P3D, 2);
  smooth(4);
  setupSerial();
  setupModels();
  loadImages();
  fira = createFont("fira.ttf", 36); // Load Fira font
  thread("runGarbageCollector");
  thread("monitorImages");
}

void setupSerial() {
  println("available ports:");
  println((Object[]) Serial.list());
  portName = Serial.list()[4];
  myPort = new Serial(this, portName, 115200);
  myPort.bufferUntil('\n');
}

void setupModels() {
  coolvetica = createFont("coolvetica rg.otf", 256);
  wrench = loadShape("wrench.obj");
  if (wrench == null) {
    println("Error loading wrench.obj");
    exit();
  }
  wrench.scale(15);
}

void loadImages() {
  synchronized (this) {
    isReloadingImages = true;
    try {
      images = new PImage[imagePaths.length];
      lastModifiedTimes = new long[imagePaths.length];
      for (int i = 0; i < imagePaths.length; i++) {
        images[i] = loadImage(imagePaths[i]);
        File file = new File(dataPath(imagePaths[i]));
        lastModifiedTimes[i] = file.lastModified();
      }
    } catch (Exception e) {
      println("Error loading images: " + e.getMessage());
    } finally {
      isReloadingImages = false;
    }
  }
}

void monitorImages() {
  while (true) {
    boolean imagesChanged = false;
    for (int i = 0; i < imagePaths.length; i++) {
      File file = new File(dataPath(imagePaths[i]));
      long lastModified = file.lastModified();
      if (lastModified > lastModifiedTimes[i]) {
        imagesChanged = true;
        lastModifiedTimes[i] = lastModified;
      }
    }
    if (imagesChanged) {
      delay(1000); // Wait for 1 second before reloading images
      loadImages();
    }
    delay(1000); // Check for changes every second
  }
}

void draw() {
  background(#FFFFFF);
  drawActions();
  displayImages();
  draw3D();
  draw2D();
  checkMovementStop();
}

// 3D effects
void draw3D() {
  pushMatrix();
  translate(width * 3 / 4, height / 4);

  ambientLight(150, 150, 150);
  pointLight(255, 255, 255, width * 3 / 4, height / 4, 200);
  lights();

  rotateX(radians(pitch));
  rotateY(radians(yaw));
  rotateZ(radians(roll));

  shape(wrench);
  popMatrix();
}

// 2D dashboard
void draw2D() {
  fill(#FFAA5A);
  drawFrame(0, 80, width / 2, height - 80, 40);
  rect(0, 0, width / 2, 80);
  fill(#6C7EA4);
  drawFrame(width / 2, 0, width / 2, height / 2, 40);
  fill(#E5E5E5);
  drawFrame(width / 2, height / 2, width / 2, height / 2, 40);

  textFont(coolvetica);
  fill(0);
  textSize(48);
  textAlign(LEFT, TOP);
  text("Position", 60, 120);
  text("Actions", width / 2 + 60, height / 2 + 60);  // Moved down by 20 pixels
  text("Rotation", width / 2 + 60, 40);
  fill(255);
  textSize(84);
  text("Fix Your Fiets Dashboard", 40, 8);
}

float scl = 75;
void displayImages() {
  synchronized (this) {
    if (!isReloadingImages) {
      for (int i = 0; i < images.length; i++) {
        if (i == 0) {
          image(images[i], 80, 200, 800, 600);
        } else {
          image(images[i], 40 + i * 4 * scl - 4 * scl, 800, 4 * scl, 3 * scl);
        }
      }
    }
  }
}

void drawFrame(int x, int y, int w, int h, int t) {
  noStroke();
  rect(x, y, t, h);
  rect(x, y, w, t);
  rect(x + w, y, -t, h);
  rect(x, y + h, w, -t);
}

void runGarbageCollector() {
  while (true) {
    delay(60000); // Run garbage collector every 60 seconds
    System.gc();
  }
}

void serialEvent(Serial myPort) {
  inString = myPort.readStringUntil('\n');
  if (inString != null) {
    inString = trim(inString);
    String[] values = split(inString, ',');
    if (values.length == 10) {
      float gyroX = float(values[3]) - 16.92060060891226;
      float gyroY = float(values[4]) - 5.1918737890949345;
      float gyroZ = float(values[5]) - (-0.013418212012178245);

      // Integrate the gyro data to get angles
      yaw += gyroZ * deltaTime;
      pitch += gyroY * deltaTime;
      roll += gyroX * deltaTime;

      checkMovement(gyroX, gyroY, gyroZ);
    }
  }
}

void checkMovement(float gyroX, float gyroY, float gyroZ) {
  float magnitude = sqrt(gyroX * gyroX + gyroY * gyroY + gyroZ * gyroZ);
  String timestamp = nf(hour(), 2) + ":" + nf(minute(), 2) + ":" + nf(second(), 2);

  if (magnitude > threshold && !isMoving) {
    actions.add(0, timestamp + " started moving");
    isMoving = true;
    checkForStop = false; // Cancel any pending stop check
  } else if (magnitude < threshold && isMoving && !checkForStop) {
    checkForStop = true;
    stopStartTime = millis();
  }

  if (actions.size() > 14) {
    actions.remove(actions.size() - 1);
  }
}

void checkMovementStop() {
  if (checkForStop && millis() - stopStartTime > 3000) {
    String timestamp = nf(hour(), 2) + ":" + nf(minute(), 2) + ":" + nf(second(), 2);
    actions.add(0, timestamp + " stopped moving");
    isMoving = false;
    checkForStop = false;

    if (actions.size() > 14) {
      actions.remove(actions.size() - 1);
    }
  }
}

void drawActions() {
  textFont(fira); // Use Fira font for actions panel
  textSize(36);
  textAlign(LEFT, TOP);
  for (int i = 0; i < actions.size(); i++) {
    float alpha = map(i, 0, actions.size(), 255, 50);
    fill(30, 30, 30, alpha);
    text(actions.get(i), width / 2 + 60, height / 2 + 120 + i * 45); // Moved down by 20 pixels
  }
}
