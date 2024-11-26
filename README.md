This is a semi-classiscal simulation for the dissociation and recombination of bottom quark states.

Current Files:
	core.py - contains the core classes: quark, bound, box
	mathFunc2.py - contains all the general math functions: rate definitions, integration routines, general math functions, etc.
	momentumSampling.py - contains event momentum sampling routines
	config.py - contains the config class that holds reverenced simulation variables as well as other "non-math" functions
Notebooks:
	mainScript.ipynb - This is the script for running the full box simulation
	CA2.ipynb - This script was for doing the channel approximations and integrations for expressing rates as functions of p
	mainScriptClean.ipynb - This is just a cleaned up version of the main script

Notes on running:

First the config gets initialized from the specified param file. All of the simulation variables, settings, and interpolations get stored here
All of the initial calculations (do integrations, set rate interpolations) get done by doConfigCalc.
^^^These two steps generally always need done first
At this point the box can be initialized and run and the numerical appriximations can be done.

*Naming convention:
	RGR-real gluon radiation(disso) 
	RGA-real gluon absorption(recom) 
	IDQ - Inelastic scattering with light quarks(disso) : IDQ:Inelastic Dissociation w/ Quarks
	IRQ - Inelastic scattering with light quarks(recom) : IRQ:Inelastic Recombination w/ Quarks
	IDG - Inelastic scattering with gluons(disso) : IDG:Inelastic Dissociation w/ Gluons
	IRG - Inelastic scattering with gluons(recom) : IRG:Inelastic Recombination w/ Gluons

*The currently difficult bit is mostly within mathFunc2.py in the inelastic recombination
*I also made a cleaned version of the main script mainScriptClean.ipynb, start with this one.
*I have it set at ambitious numerics right now but you can turn it down easiest by scaling the box size "L" and particle numbers to match density "Nbb" "NY" (in mainScriptClean there are some commented out reduced settings)
