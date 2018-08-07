#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

inline double sqr(double x) {
  return x * x;
}

inline double gaussian_observation_weight(double x1, double y1, double x2, double y2, double x3, double y3) {
  float term1 = 2.0 * M_PI * x1 * y1;
  //cout << "term1: " << term1 << "\n";
  term1 = 1.0 / term1;
  //cout << "term1: " << term1 << "\n";
  float subsubterm11 = sqr(x2 - x3);
  //cout << "subsubterm11: " << subsubterm11 << "\n";
  float subsubterm12 = 2 * sqr(x1);
  //cout << "subsubterm12: " << subsubterm12 << "\n";
  float subterm1 = (subsubterm11 / subsubterm12);
  //cout << "subterm1: " << subterm1 << "\n";
  float subsubterm21 = sqr(y2 - y3);
  //cout << "subsubterm21: " << subsubterm21 << "\n";
  float subsubterm22 = 2 * sqr(y1);
  //cout << "subsubterm22: " << subsubterm22 << "\n";
  float subterm2 = subsubterm21 / subsubterm22;
  //cout << "subterm2: " << subterm2 << "\n";
  float term2 = exp(-(subterm2 + subterm1));
  //cout << "term2: " << term2 << "\n";
  return term1 * term2;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 75;
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for(int i = 1; i <= num_particles; i++) {
    Particle new_particle;
    new_particle.id = i;
    new_particle.x = dist_x(gen);
    new_particle.y = dist_y(gen);
    new_particle.theta = dist_theta(gen);
    new_particle.weight = 1.0;
    new_particle.associations = std::vector<int>();
    new_particle.sense_x = std::vector<double>();
    new_particle.sense_y = std::vector<double>();
    particles.push_back(new_particle);
    weights.push_back(1.0);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  
  for(std::vector<Particle>::iterator i = particles.begin(); i != particles.end(); ++i) {
    Particle particle = *i;
    if(fabs(yaw_rate) >= 0.00001) {
      particle.x += (velocity/yaw_rate) * (sin(particle.theta + (yaw_rate * delta_t)) - sin(particle.theta));
      particle.y += (velocity/yaw_rate) * (-cos(particle.theta + (yaw_rate * delta_t)) + cos(particle.theta));
      particle.theta += yaw_rate * delta_t;
    } else {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    }
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
    *i = particle;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for(std::vector<LandmarkObs>::iterator i = observations.begin(); i != observations.end(); ++i) {
    LandmarkObs observation = *i;
    double least_distance = dist(predicted[0].x, predicted[0].y, observation.x, observation.y);
    for(std::vector<LandmarkObs>::iterator i2 = predicted.begin(); i2 != predicted.end(); ++i2) {
      LandmarkObs prediction = *i2;
      double new_dist = dist(prediction.x, prediction.y, observation.x, observation.y);
      if(new_dist <= least_distance) {
        least_distance = new_dist;
        i->id = prediction.id;
      }
      observation.id = prediction.id;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  for(unsigned int i = 0; i <= particles.size(); ++i) {
    Particle particle = particles[i];
    std::vector<LandmarkObs> map_coord_observations = std::vector<LandmarkObs>();
    for(std::vector<LandmarkObs>::const_iterator i2 = observations.begin(); i2 != observations.end(); ++i2) {
      LandmarkObs observation = *i2;
      double mapx = particle.x + ((cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y));
      double mapy = particle.y + ((sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y));
      
      LandmarkObs mapcoordparticle = {observation.id, mapx, mapy};
      
      map_coord_observations.push_back(mapcoordparticle);
    }
    std::vector<LandmarkObs> landmarks_close = std::vector<LandmarkObs>();
    
    for(std::vector<Map::single_landmark_s>::const_iterator i3 = map_landmarks.landmark_list.begin(); i3 != map_landmarks.landmark_list.end(); ++i3) {
      Map::single_landmark_s landmark = *i3;
      double land_part_dist = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);
      if (land_part_dist <= sensor_range) {
        LandmarkObs landmark_map = {landmark.id_i, landmark.x_f, landmark.y_f};
        landmarks_close.push_back(landmark_map);
      }
    }
    dataAssociation(landmarks_close, map_coord_observations);
    
    double observation_weight = 1;
    for(std::vector<LandmarkObs>::iterator i4 = map_coord_observations.begin(); i4 != map_coord_observations.end(); ++i4) {
      LandmarkObs matching_prediction;
      LandmarkObs map_observation = *i4;
      for(std::vector<LandmarkObs>::iterator i5 = landmarks_close.begin(); i5 != landmarks_close.end(); ++i5) {
        LandmarkObs close_landmark = *i5;
        if(close_landmark.id == map_observation.id) {
          matching_prediction = close_landmark;
          break;
        }
      }
      float observation_weight_multiplier = gaussian_observation_weight(std_landmark[0], std_landmark[1], map_observation.x, map_observation.y, matching_prediction.x, matching_prediction.y);
      observation_weight *= observation_weight_multiplier;
    }
    weights[i] = observation_weight;
    particles[i].weight = observation_weight;
  }
  double weight_total = 0;
  for(int i = 0; i <= (num_particles - 1); i++) {
    weight_total += weights[i];
  }
  for(int i = 0; i <= (num_particles - 1); i++) {
    if(weight_total <= 0.00001) {
      weights[i] = 1.0 / weight_total;
      particles[i].weight = 1.0 / weight_total;
      break;
    } else {
      weights[i] /= weight_total;
      particles[i].weight /= weight_total;
    }
 
  }
}

void ParticleFilter::resample() {
  std::default_random_engine rand_generator;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles;
  
  while(resampled_particles.size() < particles.size()) {
    resampled_particles.push_back(particles[d(rand_generator)]);
  }
  
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
