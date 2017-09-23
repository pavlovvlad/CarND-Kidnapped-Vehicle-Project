/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;

    // create rundom engine
	default_random_engine gen;

    // create normal distribution for position and orientation of particles
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    // reserve the memory to avoid reallocation by each pushback, since the size is known
    particles.reserve(num_particles);
    weights.reserve(num_particles);

	// Initialize all particles to first position with added gaussian noise
	for(int i=0; i<num_particles; ++i){
	  Particle temp = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
      particles.push_back(temp);
      weights.push_back(1.0);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// define the very small value for a float-to-zero comparision 
    double eps = 1e8;

	// create rundom engine
	default_random_engine gen;
   
	// create normal distribution for position and orientation of particles
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std[2]);
    
    // predict the states with the CTRV model
    for (int i = 0; i < particles.size(); ++i) {

      double theta_p = particles[i].theta + yaw_rate * delta_t;
    
      // avoid division by zero
      if (eps < fabs(yaw_rate)) {
        particles[i].x += velocity / yaw_rate * (sin(theta_p) - sin(particles[i].theta));
        particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(theta_p));
      }
      else { // no rotation
        particles[i].x += velocity * cos(particles[i].theta) * delta_t;
        particles[i].y += velocity * sin(particles[i].theta) * delta_t;
      }

      // assign the new theta with noise
      particles[i].theta = theta_p + dist_theta(gen);

      // add noise in position
      particles[i].x += dist_x(gen);
      particles[i].y += dist_y(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); ++i){
        
        double nearest_dist = 1e8;
        int nearest_id = 0;

        for (int j = 0; j < predicted.size(); ++j){

            // Euclidian distance in cartesian coordinates
            double dist_ = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            if (dist_ < nearest_dist){

                nearest_id = predicted[j].id;
                nearest_dist = dist_;
            }
        }

        observations[i].id = nearest_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    double var_x = pow(std_landmark[0], 2);
    double var_y = pow(std_landmark[1], 2);

	double gauss_norm = 0.5/(M_PI*std_landmark[0]*std_landmark[1]);

	// for each particle 
    for (int i = 0; i < particles.size(); ++i){

        // find the landmarks in the FOV of the sensor (sensor_range) for current particle
        vector<LandmarkObs> landmarks_sensed;
        for (int j = 0; j < map_landmarks.landmark_list.size(); ++j){

            Map::single_landmark_s landmark_j = map_landmarks.landmark_list[j];

            double dist_j = dist(particles[i].x, particles[i].y, landmark_j.x_f, landmark_j.y_f);

            if (sensor_range > dist_j){

                landmarks_sensed.push_back({landmark_j.id_i, landmark_j.x_f, landmark_j.y_f});
            }
        }

        // transform observations to x,y-coordinates relative to particle position
        vector<LandmarkObs> obs_map(observations.size());
        
        for (int j = 0; j < observations.size(); ++j){

            obs_map[j].id = observations[j].id;
            obs_map[j].x = particles[i].x + observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta);
            obs_map[j].y = particles[i].y + observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta);
        }

        // find the predicted measurement that is closest to each observed measurement and assign the 
	    // observed measurement to this particular landmark
        dataAssociation(landmarks_sensed, obs_map);
        
        // init temp weight
        double current_weight = 1.0;
        
        // update weights        
        for (int k = 0; k < obs_map.size(); ++k){

            double mu_x = 0.0;
            double mu_y = 0.0;

            for (int j = 0; j < landmarks_sensed.size(); ++j){

                if (landmarks_sensed[j].id == obs_map[k].id){

                    mu_x = landmarks_sensed[j].x;
                    mu_y = landmarks_sensed[j].y;
                }
            }
            // calculate weight using normalization terms and exponent
            current_weight *= gauss_norm*exp(-0.5*(pow(obs_map[k].x - mu_x, 2)/var_x + pow(obs_map[k].y - mu_y, 2)/var_y));

        }

        particles[i].weight = current_weight;
        weights[i] = current_weight; 
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<int> discrete_dist(weights.begin(), weights.end());

    vector<Particle> resampled_particles(particles.size());

    for (int i = 0; i < particles.size(); ++i){
        
        resampled_particles[i] = particles[discrete_dist(gen)];
        weights[i] = particles[discrete_dist(gen)].weight;

    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
