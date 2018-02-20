/*
 * ntuple_content.h
 *
 *  Created on: 3 Jun 2017
 *      Author: jkiesele
 */

#ifndef INCLUDE_NTUPLE_CONTENT_H_
#define INCLUDE_NTUPLE_CONTENT_H_

#include "TTree.h"
#include "TString.h"

class NTupleContent{
public:

	NTupleContent():read_(false){
		//
	}

	virtual ~NTupleContent(){}

	virtual void reset()=0;

	void setIsRead(bool isread){read_=isread;}
	virtual void initDNNBranches(TTree* )=0;

	void initBranches(TTree* t){
		initDNNBranches(t);//compatibility with merging from deepNtuples
	}



protected:
	template <class T>
	void addBranch(TTree* t, const char* name,  T*, const char* leaflist=0);

private:

	bool read_;
	std::vector<TString> allbranches_;

};

template <class T>
void NTupleContent::addBranch(TTree* t, const char* name,  T* address, const char* leaflist){

	if(read_ ){
		t->SetBranchAddress(name,address);
	}
	else{
		if(leaflist)
			t->Branch(name  ,address  ,leaflist );
		else
			t->Branch(name  ,address);
	}
	allbranches_.push_back((TString)name);

}





#endif /* INCLUDE_NTUPLE_CONTENT_H_ */
