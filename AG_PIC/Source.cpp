#include <SFML/Graphics.hpp>
#include <SFML\Graphics\Image.hpp>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <Windows.h>
#include <ctime>
#include <fstream>

using namespace sf;
using namespace std;

#define CIBLE "images/target8.jpg"
#define NB_POP 1
//#define RAPIDE
#define HIDE
#define RAND_COL
//#define SAVE

#define ALPHA 0.01f
#define EX_FICH "exemples8.txt"
#define NB_ENTREES 3
#define NB_NEURONES 8
#define NB_POP_SLP 1000
#define GENERATION_MIN 150

enum Fonction{
	HEAVYSIDE, LINEAR, SIGMOIDE
};

enum Color_SLP{
	RED = 1, GREEN = 2, BLUE = 3,BLACK, WHITE,YELLOW = 6,CYAN,MAGENTA
};

struct Perceptron;

struct Neurone{
	Perceptron* pere;
	float weight[NB_ENTREES];
	float bias;
	float output;
	float(*f)(float, float);
};

struct ColorStr{
	float r, v, b;
	Color_SLP out;
};

struct Perceptron{
	float input[NB_ENTREES];
	float output[NB_NEURONES];
	vector<shared_ptr<Neurone>> neurones;
};

struct Individu{
	float W[NB_NEURONES][NB_ENTREES]; //Poids [Neurone][entree]
	float B[NB_NEURONES]; //Biais[Neurone]
	float fitness; //fitness = somme
};

bool heaviside(float sum, float bias){
	return sum >= bias;
}

float linear(float sum, float bias){
	if (sum / 255 >= 1){
		return 1;
	}
	else if (sum / 255 <= 0){
		return 0;
	}
	else{
		return sum / 255;
	}
}

float sigmoide(float sum, float bias){
	float a = expf(-sum / bias);
	return 1 / (1 + a);
}

void initNeurone(Neurone* N, Perceptron* p){
	N->pere = p;
	N->bias = rand() % 101;
	N->bias /= 100;
	for (int i = 0; i < NB_ENTREES; i++){
		N->weight[i] = rand() % 101;
		N->weight[i] /= 100;
	}
	N->f = linear;
}

void change(Neurone* n){
	for (int i = 0; i < NB_ENTREES; i++){
		n->weight[i] += ALPHA;
		if (n->weight[i] > 1){
			n->weight[i] = 0;
		}
	}
	n->bias += ALPHA;
	if (n->bias > 1){
		n->bias = 0;
	}
}

float eval(Neurone* n){
	float a = 0;
	for (int i = 0; i < NB_ENTREES; i++){
		a += n->pere->input[i] * n->weight[i];
	}
	return a;
}

void initExemples(vector<ColorStr>* ex, ifstream* flx){
	int i = 0;
	int j = 0;
	while (!flx->eof()){
		switch (i)
		{
		case 0:
			ex->resize(ex->size() + 1);
			*flx >> ex->at(j).r;
			break;
		case 1:
			*flx >> ex->at(j).v;
			break;
		case 2:
			*flx >> ex->at(j).b;
			break;
		case 3:
			int k;
			*flx >> k;
			ex->at(j).out = static_cast<Color_SLP>(k);
			i = -1;
			j++;
			break;
		default:
			break;
		}
		i++;
	}
}

void appliquerIndivAPerceptron(Perceptron* p, Individu& indiv){
	for (int i = 0; i < p->neurones.size(); i++){
		p->neurones.at(i)->bias = indiv.B[i];
		for (int j = 0; j < NB_ENTREES; j++){
			p->neurones.at(i)->weight[j] = indiv.W[i][j];
		}
	}

}

void initInputs(Perceptron* p, vector<ColorStr>& exemples, int nb_essais){
	p->input[0] = exemples.at(nb_essais).r;
	p->input[1] = exemples.at(nb_essais).v;
	p->input[2] = exemples.at(nb_essais).b;
}

void setInputs(Perceptron* p, int r, int g, int b){
	p->input[0] = r;
	p->input[1] = g;
	p->input[2] = b;
}

void calcOutputs(Perceptron* p){
	for (int u = 0; u < NB_NEURONES; u++){
		float e = eval(p->neurones.at(u).get());
		float s = p->neurones.at(u)->f(e, p->neurones.at(u)->bias);
		p->output[u] = s;
	}
}

int findMax(Perceptron* p){
	int idm = 0;
	for (int r = 1; r < NB_NEURONES; r++){
		if (p->output[r] > p->output[idm]){
				idm = r; 
		}
	}
	idm++;
	return idm;
}

void initIndiv(shared_ptr<Individu>* indiv){
	indiv->reset(new Individu());
	for (int i = 0; i < NB_NEURONES; i++){
		for (int j = 0; j < NB_ENTREES; j++){
			indiv->get()->W[i][j] = rand() % 101;
			indiv->get()->W[i][j] /= 100;
		}
		indiv->get()->B[i] = rand() % 101;
		indiv->get()->B[i] /= 100;
	}
	indiv->get()->fitness = -1;
}

void initPop_SLP(vector<shared_ptr<Individu>>* pop){
	for (int i = 0; i < pop->size(); i++){
		initIndiv(&pop->at(i));
	}
}

int findBestIndiv(vector<shared_ptr<Individu>>* pop){
	int idmin = 0;
	for (int i = 1; i < pop->size(); i++){
		if (pop->at(idmin)->fitness > pop->at(i)->fitness){
			idmin = i;
		}
	}
	return idmin;
}

void mutation_SLP(shared_ptr<Individu>* indiv){
	if (rand() % 2 == 1){
		int x = rand() % NB_NEURONES;
		int y = rand() % NB_ENTREES;
		indiv->get()->W[x][y] = rand() % 101;
		indiv->get()->W[x][y] /= 100;

	}
	else{
		int x = rand() % NB_NEURONES;
		indiv->get()->B[x] = rand() % 101;
		indiv->get()->B[x] /= 100;

	}
}

void affInfiv(Individu& best){
	for (int i = 0; i < NB_NEURONES; i++){
		for (int j = 0; j < NB_ENTREES; j++){
			cout << best.W[i][j] << " ";
		}
		cout << endl;
	}
	for (int j = 0; j < NB_NEURONES; j++){
		cout << best.B[j] << " ";
	}
	cout << endl;
}

bool trieFonct_SLP(shared_ptr<Individu> i1, shared_ptr<Individu> i2){
	return i1->fitness < i2->fitness;
}

void trie(vector<shared_ptr<Individu>>* pop){
	sort(pop->begin(), pop->end(), trieFonct_SLP);
}

void selectParents_SLP(vector<shared_ptr<Individu>>* pop, pair<int, int>* parents){
	/*int r = rand() % (int(pop->back()->fitness)+1);
	int i = 0;
	while (pop->at(i)->fitness > r && i < NB_POP-1){
	i++;
	}
	parents->first = i;
	r = rand() % (int(pop->back()->fitness)+1);
	i = 0;
	while (pop->at(i)->fitness > r && i < NB_POP - 1){
	i++;
	}
	parents->second = i;*/

	parents->first = rand() % (pop->size() / (NB_POP_SLP / 100));
	parents->second = rand() % (pop->size() / (NB_POP_SLP/100));

}

Individu child_SLP(vector<shared_ptr<Individu>>* pop){
	pair<int, int> parents;
	selectParents_SLP(pop, &parents);
	Individu child;
	for (int i = 0; i < NB_NEURONES; i++){
		for (int j = 0; j < NB_ENTREES; j++){
			child.W[i][j] = pop->at(parents.first)->W[i][j];
		}
		child.B[i] = pop->at(parents.second)->B[i];
	}
	return child;
}


void newPop_SLP(vector<shared_ptr<Individu>>* pop){
	vector<shared_ptr<Individu>> buffer(pop->size());
	for (int i = 0; i < pop->size(); i++){
		buffer.at(i) = shared_ptr<Individu>(new Individu(child_SLP(pop)));
		//buffer.at(i)->fitness = pop->at(i)->fitness;
	}
	*pop = buffer;
}

void reinjection_SLP(vector<shared_ptr<Individu>>* pop, Individu best){
	int idmax = 0;
	for (int i = 1; i < pop->size(); i++){
		if (pop->at(i)->fitness > pop->at(idmax)->fitness){
			idmax = i;
		}
	}
	pop->at(idmax).reset(new Individu(best));
}

//FIN PERCEPTRON

void  initPop(vector<shared_ptr<Image>>* pop, Image* cible, Perceptron* p){
	for (int i = 0; i < pop->size(); i++){
		pop->at(i).reset(new Image());
		pop->at(i)->create(cible->getSize().x, cible->getSize().y);
		for (int j = 0; j < cible->getSize().x; j++){
			//pop->at(i)->setPixel(j, ligne, Color::White);
			for (int k = 0; k < cible->getSize().y; k++){
#ifdef RAND_COL
				Color c(rand() % 256, rand() % 256, rand() % 256, rand() % 256);
				setInputs(p, cible->getPixel(j, k).r, cible->getPixel(j, k).g, cible->getPixel(j, k).b);
				calcOutputs(p);
				switch (findMax(p))
				{
				case 1:
					pop->at(i)->setPixel(j, k, Color::Red);
					break;
				case 2:
					pop->at(i)->setPixel(j, k, Color::Green);
					break;
				case 3:
					pop->at(i)->setPixel(j, k, Color::Blue);
					break;
				case 4:
					pop->at(i)->setPixel(j, k, Color::Black);
					break;
				case 5:
					pop->at(i)->setPixel(j, k, Color::White);
					break;
				case 6:
					pop->at(i)->setPixel(j, k, Color::Yellow);
					break;
				case 7:
					pop->at(i)->setPixel(j, k, Color::Cyan);
					break;
				case 8:
					pop->at(i)->setPixel(j, k, Color::Magenta);
					break;
				default:
					break;
				}
				//pop->at(i)->setPixel(j, k, c);
#endif
#ifdef WHITE
				pop->at(i)->setPixel(j, k, Color::White);
#endif
			}
		}
	}
}

int fitIndiv(shared_ptr<Image> indiv, Image* cible){
	int somme = 0;
	for (int i = 0; i < cible->getSize().x; i++){
		for (int ligne = 0; ligne < cible->getSize().y; ligne++){
			somme += abs(cible->getPixel(i, ligne).r - indiv->getPixel(i, ligne).r);
			somme += abs(cible->getPixel(i, ligne).g - indiv->getPixel(i, ligne).g);
			somme += abs(cible->getPixel(i, ligne).b - indiv->getPixel(i, ligne).b);
		}

	}
	return somme;
}
void fitPop(vector<int>* fit, vector<shared_ptr<Image>>* pop, Image* cible){
	for (int i = 0; i < pop->size(); i++){
		fit->at(i) = fitIndiv(pop->at(i), cible);
	}
}

int findBest(vector<int>* fit){
	int id = 0;
	for (int i = 1; i < fit->size(); i++){
		if (fit->at(i) < fit->at(id)){
			id = i;
		}
	}
	return id;
}

bool trieFonct(pair<int,int> a, pair<int,int> b){
	return a.first > b.first;
}

void trieFit(vector<pair<int,int>>* fitTrie, vector<int>* fit){
	for (int i = 0; i < fit->size(); i++){
		fitTrie->at(i).first = fit->at(i);
		fitTrie->at(i).second = i;
	}
	sort(fitTrie->begin(), fitTrie->end(), trieFonct);
}

pair<int, int> selectParents(vector<pair<int, int>>* fitTrie){
	pair<int, int> parents;
	int r = rand() % (fitTrie->back().first+1);
	//r += 1;
	int i = 0;
	while (r < fitTrie->at(i).first && i < NB_POP-1){
		i++;
	}
	parents.first = fitTrie->at(i).second;
	//do{
		r = rand() % (fitTrie->back().first + 1);
		//r += 1;
		i = 0;
		while (r < fitTrie->at(i).first && i < NB_POP-1){
			i++;
		}
		parents.second = fitTrie->at(i).second;
	//} while (parents.first == parents.second);
	return parents;
}

void mutation(shared_ptr<Image> ch,Image* cible){
	int size = ch->getSize().x * ch->getSize().y ;
	int r = rand() % size;
	do{
		Color c(rand() % 256, rand() % 256, rand() % 256);
		int x = rand() % ch->getSize().x;
		int y = rand()%ch->getSize().y;
#ifdef RAPIDE
		if (ch->getPixel(x, y) != cible->getPixel(x, y)){
			ch->setPixel(x, y, cible->getPixel(x,y));//SOLUTION DE FACILITE
			//ch->setPixel(x, y, c);//VRAIE SOLUTION
		}
#endif
#ifndef RAPIDE
		//SOLUTION CAS PAR CAS
		if (ch->getPixel(x, y) != cible->getPixel(x, y)){
			if (ch->getPixel(x, y).r != cible->getPixel(x, y).r){
				int a = abs(ch->getPixel(x, y).r - cible->getPixel(x, y).r);
				ch->setPixel(x, y, *shared_ptr<Color>(new Color(ch->getPixel(x, y).r+(rand() % 2 * a) - a, ch->getPixel(x, y).g, ch->getPixel(x, y).b)));
			}
			if (ch->getPixel(x, y).g != cible->getPixel(x, y).g){
				int a = abs(ch->getPixel(x, y).g - cible->getPixel(x, y).g);
				ch->setPixel(x, y, *shared_ptr<Color>(new Color(ch->getPixel(x, y).r, ch->getPixel(x, y).g+(rand() % 2 * a) - a, ch->getPixel(x, y).b)));
			}
			if (ch->getPixel(x, y).b != cible->getPixel(x, y).b){
				int a = abs(ch->getPixel(x, y).b - cible->getPixel(x, y).b);
				ch->setPixel(x, y, *shared_ptr<Color>(new Color(ch->getPixel(x, y).r, ch->getPixel(x, y).g, ch->getPixel(x, y).b+(rand() % 2 * a) - a)));
			}
		}
#endif

		
		
		r--;
	} while (r > 0);
}

shared_ptr<Image> child(vector<pair<int, int>>* fitTrie, vector<shared_ptr<Image>>* pop,Image* cible){
	pair<int, int> parents = selectParents(fitTrie);
	shared_ptr<Image> child;
	child.reset(new Image());
	child->create(cible->getSize().x, cible->getSize().y);
	//child->copy(*pop->at(parents.first), 0, 0);
	for (int i = 0; i < cible->getSize().x / 2; i++){
		for (int j = 0; j < cible->getSize().y; j++){
			child->setPixel(i, j, pop->at(parents.first)->getPixel(i, j));
		}
	}
	//child->copy(*pop->at(parents.second), 0, child->getSize().y/2);
	for (int i = cible->getSize().x / 2; i < cible->getSize().x; i++){
		
		for (int j = 0; j < cible->getSize().y; j++){
			child->setPixel(i, j, pop->at(parents.second)->getPixel(i, j));
		}
		
	}
	mutation(child,cible);
	return child;
}

void newPop(vector<pair<int, int>>* fitTrie, vector<shared_ptr<Image>>* pop, Image* cible){
	vector<shared_ptr<Image>> buffer(NB_POP);
	for (int i = 0; i < buffer.size(); i++){
		buffer.at(i) = child(fitTrie, pop, cible);
	}
	*pop = buffer;
}

void reinjection(shared_ptr<Image> best, vector<shared_ptr<Image>>* pop, vector<int>* fit){
	int id = 0;
	for (int i = 1; i < fit->size(); i++){
		if (fit->at(i) > fit->at(id)){
			id = i;
		}
	}
	pop->at(id) = best;
}

int main()
{	
	int t = time(NULL);
	srand(time(NULL));


	vector<ColorStr> exemples;
	ifstream fich(EX_FICH, ios::in);
	if (fich){
		initExemples(&exemples, &fich);
	}
	else{
		return 1;
	}

	vector<shared_ptr<Individu>> population_SLP(NB_POP_SLP);
	Individu best_SLP;
	initPop_SLP(&population_SLP);
	best_SLP = *population_SLP.at(0).get();
	best_SLP.fitness = 100;
	Perceptron p;
	for (int i = 0; i < NB_NEURONES; i++){
		p.neurones.push_back(shared_ptr<Neurone>(new Neurone));
		initNeurone(p.neurones.at(i).get(), &p);
	}
	int somme = -1;
	int generation = 0;
	int nb_essais = 0;
	while (best_SLP.fitness > 0 && generation < GENERATION_MIN){
		//cout <<"GENERATION : " <<generation << endl;
		for (int i = 0; i < population_SLP.size(); i++){
			somme = 0;
			nb_essais = 0;
			appliquerIndivAPerceptron(&p, *population_SLP.at(i).get());
			while (nb_essais < exemples.size()){
				initInputs(&p, exemples, nb_essais);
				calcOutputs(&p);
				int d = abs(exemples.at(nb_essais).out - findMax(&p));
				if (d != 0){
					//cout << "ligne : " << nb_essais << endl;
				}
				somme += d;
				nb_essais++;
			}
			population_SLP.at(i)->fitness = somme;

		}
		if (population_SLP.at(findBestIndiv(&population_SLP)).get()->fitness < best_SLP.fitness || population_SLP.at(findBestIndiv(&population_SLP)).get()->fitness == 0){
			best_SLP = *population_SLP.at(findBestIndiv(&population_SLP)).get();
			cout << best_SLP.fitness << endl;
		}
		trie(&population_SLP);
		reinjection_SLP(&population_SLP, best_SLP);
		newPop_SLP(&population_SLP);
		for (int i = 0; i < population_SLP.size(); i++){
			mutation_SLP(&population_SLP.at(i));
		}

		//selectParents+reinj


		//_sleep(100);
		generation++;
	}
	int r, g, b;
	char f = ' ';
	affInfiv(best_SLP);
	appliquerIndivAPerceptron(&p, best_SLP);







	Image cible;
	cible.loadFromFile(CIBLE);
#ifdef SHOW
    sf::RenderWindow window(sf::VideoMode(cible.getSize().x*2,cible.getSize().y), "SFML works!");
#endif

#ifdef HIDE
	sf::RenderWindow window(sf::VideoMode(cible.getSize().x , cible.getSize().y), "SFML works!");
#endif

	int gen = 0;
	vector<shared_ptr<Image>> population(NB_POP);
	shared_ptr<Image> best;
	best.reset(new Image());
	best->create(cible.getSize().x, cible.getSize().y);
	initPop(&population, &cible,&p);
	vector<int> fitpop(NB_POP);
	vector<pair<int,int>> fitTrie(NB_POP);
	fitPop(&fitpop, &population, &cible);
	best->copy(*population.at(findBest(&fitpop)),0,0);
	

	Texture texture,texC;
	texture.loadFromImage(*best);
	texC.loadFromImage(cible);
	Sprite sprite,spC;
	sprite.setTexture(texture);
	spC.setTexture(texC);
	spC.setPosition(cible.getSize().x,0);
	

    while (window.isOpen() && fitIndiv(best,&cible) != 0)
    {
		cout << gen << endl;
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
		trieFit(&fitTrie, &fitpop);
		best->copy(*population.at(findBest(&fitpop)), 0, 0);
		texture.loadFromImage(*best);
		window.draw(sprite);
#ifdef SHOW
		window.draw(spC);
#endif
		window.display();
		newPop(&fitTrie, &population, &cible);
		mutation(best, &cible);
		reinjection(best, &population, &fitpop);
		gen++;
		cout << time(NULL) - t << "s "  << "[" << fitIndiv(best,&cible) << "]" << endl;
		fitPop(&fitpop, &population, &cible);
#ifdef SAVE
		if (gen % 10 == 0 || gen == 1){
			
			string name = CIBLE + to_string(gen) + ".jpg";
			best->saveToFile(name);
		}
#endif
    }
	
	system("PAUSE");
	fich.close();
    return 0;
}