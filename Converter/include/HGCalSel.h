//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Sep  7 17:22:11 2017 by ROOT version 6.04/03
// from TTree hgc/hgc
// found on file: /afs/cern.ch/user/j/jkiesele/eos_hgcal/FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_eta_2.3to2.5_timing_20170907/NTUP/partGun_PDGid11_x960_Pt2.0To100.0_NTUP_92.root
//////////////////////////////////////////////////////////

#ifndef HGCalSel_h
#define HGCalSel_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include <vector>

class HGCalSel {
public :
    TTree *fChain;   //!pointer to the analyzed TTree or TChain
    Int_t fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

    // Declaration of leaf types
    ULong64_t event;
    UInt_t lumi;
    UInt_t run;
    Float_t vtx_x;
    Float_t vtx_y;
    Float_t vtx_z;
    std::vector<float> *genpart_eta;
    std::vector<float> *genpart_phi;
    std::vector<float> *genpart_pt;
    std::vector<float> *genpart_energy;
    std::vector<float> *genpart_dvx;
    std::vector<float> *genpart_dvy;
    std::vector<float> *genpart_dvz;
    std::vector<float> *genpart_ovx;
    std::vector<float> *genpart_ovy;
    std::vector<float> *genpart_ovz;
    std::vector<int> *genpart_mother;
    std::vector<float> *genpart_exphi;
    std::vector<float> *genpart_exeta;
    std::vector<float> *genpart_exx;
    std::vector<float> *genpart_exy;
    std::vector<float> *genpart_fbrem;
    std::vector<int> *genpart_pid;
    std::vector<int> *genpart_gen;
    std::vector<int> *genpart_reachedEE;
    std::vector<bool> *genpart_fromBeamPipe;
    std::vector<std::vector<float> > *genpart_posx;
    std::vector<std::vector<float> > *genpart_posy;
    std::vector<std::vector<float> > *genpart_posz;
    std::vector<float> *rechit_eta;
    std::vector<float> *rechit_phi;
    std::vector<float> *rechit_pt;
    std::vector<float> *rechit_energy;
    std::vector<float> *rechit_x;
    std::vector<float> *rechit_y;
    std::vector<float> *rechit_z;
    std::vector<float> *rechit_time;
    std::vector<float> *rechit_thickness;
    std::vector<int> *rechit_layer;
    std::vector<int> *rechit_wafer;
    std::vector<int> *rechit_cell;
    std::vector<unsigned int> *rechit_detid;
    std::vector<bool> *rechit_isHalf;
    std::vector<int> *rechit_flags;
    std::vector<int> *rechit_cluster2d;
    std::vector<float> *cluster2d_eta;
    std::vector<float> *cluster2d_phi;
    std::vector<float> *cluster2d_pt;
    std::vector<float> *cluster2d_energy;
    std::vector<float> *cluster2d_x;
    std::vector<float> *cluster2d_y;
    std::vector<float> *cluster2d_z;
    std::vector<int> *cluster2d_layer;
    std::vector<int> *cluster2d_nhitCore;
    std::vector<int> *cluster2d_nhitAll;
    std::vector<int> *cluster2d_multicluster;
    std::vector<std::vector<unsigned int> > *cluster2d_rechits;
    std::vector<int> *cluster2d_rechitSeed;
    std::vector<float> *multiclus_eta;
    std::vector<float> *multiclus_phi;
    std::vector<float> *multiclus_pt;
    std::vector<float> *multiclus_energy;
    std::vector<float> *multiclus_z;
    std::vector<float> *multiclus_slopeX;
    std::vector<float> *multiclus_slopeY;
    std::vector<std::vector<unsigned int> > *multiclus_cluster2d;
    std::vector<int> *multiclus_cl2dSeed;
    std::vector<int> *multiclus_firstLay;
    std::vector<int> *multiclus_lastLay;
    std::vector<int> *multiclus_NLay;
    std::vector<float> *simcluster_eta;
    std::vector<float> *simcluster_phi;
    std::vector<float> *simcluster_pt;
    std::vector<float> *simcluster_energy;
    std::vector<float> *simcluster_simEnergy;
    std::vector<std::vector<unsigned int> > *simcluster_hits;
    std::vector<std::vector<float> > *simcluster_fractions;
    std::vector<std::vector<unsigned int> > *simcluster_layers;
    std::vector<std::vector<unsigned int> > *simcluster_wafers;
    std::vector<std::vector<unsigned int> > *simcluster_cells;
    std::vector<float> *pfcluster_eta;
    std::vector<float> *pfcluster_phi;
    std::vector<float> *pfcluster_pt;
    std::vector<float> *pfcluster_energy;
    std::vector<float> *pfcluster_correctedEnergy;
    std::vector<std::vector<unsigned int> > *pfcluster_hits;
    std::vector<std::vector<float> > *pfcluster_fractions;
    std::vector<float> *calopart_eta;
    std::vector<float> *calopart_phi;
    std::vector<float> *calopart_pt;
    std::vector<float> *calopart_energy;
    std::vector<float> *calopart_simEnergy;
    std::vector<std::vector<unsigned int> > *calopart_simClusterIndex;
    std::vector<float> *track_eta;
    std::vector<float> *track_phi;
    std::vector<float> *track_pt;
    std::vector<float> *track_energy;
    std::vector<int> *track_charge;
    std::vector<std::vector<float> > *track_posx;
    std::vector<std::vector<float> > *track_posy;
    std::vector<std::vector<float> > *track_posz;

    // List of branches
    TBranch *b_event;   //!
    TBranch *b_lumi;   //!
    TBranch *b_run;   //!
    TBranch *b_vtx_x;   //!
    TBranch *b_vtx_y;   //!
    TBranch *b_vtx_z;   //!
    TBranch *b_genpart_eta;   //!
    TBranch *b_genpart_phi;   //!
    TBranch *b_genpart_pt;   //!
    TBranch *b_genpart_energy;   //!
    TBranch *b_genpart_dvx;   //!
    TBranch *b_genpart_dvy;   //!
    TBranch *b_genpart_dvz;   //!
    TBranch *b_genpart_ovx;   //!
    TBranch *b_genpart_ovy;   //!
    TBranch *b_genpart_ovz;   //!
    TBranch *b_genpart_mother;   //!
    TBranch *b_genpart_exphi;   //!
    TBranch *b_genpart_exeta;   //!
    TBranch *b_genpart_exx;   //!
    TBranch *b_genpart_exy;   //!
    TBranch *b_genpart_fbrem;   //!
    TBranch *b_genpart_pid;   //!
    TBranch *b_genpart_gen;   //!
    TBranch *b_genpart_reachedEE;   //!
    TBranch *b_genpart_fromBeamPipe;   //!
    TBranch *b_genpart_posx;   //!
    TBranch *b_genpart_posy;   //!
    TBranch *b_genpart_posz;   //!
    TBranch *b_rechit_eta;   //!
    TBranch *b_rechit_phi;   //!
    TBranch *b_rechit_pt;   //!
    TBranch *b_rechit_energy;   //!
    TBranch *b_rechit_x;   //!
    TBranch *b_rechit_y;   //!
    TBranch *b_rechit_z;   //!
    TBranch *b_rechit_time;   //!
    TBranch *b_rechit_thickness;   //!
    TBranch *b_rechit_layer;   //!
    TBranch *b_rechit_wafer;   //!
    TBranch *b_rechit_cell;   //!
    TBranch *b_rechit_detid;   //!
    TBranch *b_rechit_isHalf;   //!
    TBranch *b_rechit_flags;   //!
    TBranch *b_rechit_cluster2d;   //!
    TBranch *b_cluster2d_eta;   //!
    TBranch *b_cluster2d_phi;   //!
    TBranch *b_cluster2d_pt;   //!
    TBranch *b_cluster2d_energy;   //!
    TBranch *b_cluster2d_x;   //!
    TBranch *b_cluster2d_y;   //!
    TBranch *b_cluster2d_z;   //!
    TBranch *b_cluster2d_layer;   //!
    TBranch *b_cluster2d_nhitCore;   //!
    TBranch *b_cluster2d_nhitAll;   //!
    TBranch *b_cluster2d_multicluster;   //!
    TBranch *b_cluster2d_rechits;   //!
    TBranch *b_cluster2d_rechitSeed;   //!
    TBranch *b_multiclus_eta;   //!
    TBranch *b_multiclus_phi;   //!
    TBranch *b_multiclus_pt;   //!
    TBranch *b_multiclus_energy;   //!
    TBranch *b_multiclus_z;   //!
    TBranch *b_multiclus_slopeX;   //!
    TBranch *b_multiclus_slopeY;   //!
    TBranch *b_multiclus_cluster2d;   //!
    TBranch *b_multiclus_cl2dSeed;   //!
    TBranch *b_multiclus_firstLay;   //!
    TBranch *b_multiclus_lastLay;   //!
    TBranch *b_multiclus_NLay;   //!
    TBranch *b_simcluster_eta;   //!
    TBranch *b_simcluster_phi;   //!
    TBranch *b_simcluster_pt;   //!
    TBranch *b_simcluster_energy;   //!
    TBranch *b_simcluster_simEnergy;   //!
    TBranch *b_simcluster_hits;   //!
    TBranch *b_simcluster_fractions;   //!
    TBranch *b_simcluster_layers;   //!
    TBranch *b_simcluster_wafers;   //!
    TBranch *b_simcluster_cells;   //!
    TBranch *b_pfcluster_eta;   //!
    TBranch *b_pfcluster_phi;   //!
    TBranch *b_pfcluster_pt;   //!
    TBranch *b_pfcluster_energy;   //!
    TBranch *b_pfcluster_correctedEnergy;   //!
    TBranch *b_pfcluster_hits;   //!
    TBranch *b_pfcluster_fractions;   //!
    TBranch *b_calopart_eta;   //!
    TBranch *b_calopart_phi;   //!
    TBranch *b_calopart_pt;   //!
    TBranch *b_calopart_energy;   //!
    TBranch *b_calopart_simEnergy;   //!
    TBranch *b_calopart_simClusterIndex;   //!
    TBranch *b_track_eta;   //!
    TBranch *b_track_phi;   //!
    TBranch *b_track_pt;   //!
    TBranch *b_track_energy;   //!
    TBranch *b_track_charge;   //!
    TBranch *b_track_posx;   //!
    TBranch *b_track_posy;   //!
    TBranch *b_track_posz;   //!

    HGCalSel(TTree *tree = 0);

    virtual ~HGCalSel();

    virtual Int_t Cut(Long64_t entry);

    virtual Int_t GetEntry(Long64_t entry);

    virtual Long64_t LoadTree(Long64_t entry);

    virtual void Init(TTree *tree);

    virtual void Loop();

    virtual Bool_t Notify();

    virtual void Show(Long64_t entry = -1);
};

#endif

#ifdef HGCalSel_cxx

HGCalSel::HGCalSel(TTree *tree) : fChain(0) {
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
    if (tree == 0) {
        TFile *f = (TFile *) gROOT->GetListOfFiles()->FindObject(
                "/afs/cern.ch/user/j/jkiesele/eos_hgcal/FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_eta_2.3to2.5_timing_20170907/NTUP/partGun_PDGid11_x960_Pt2.0To100.0_NTUP_92.root");
        if (!f || !f->IsOpen()) {
            f = new TFile(
                    "/afs/cern.ch/user/j/jkiesele/eos_hgcal/FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_eta_2.3to2.5_timing_20170907/NTUP/partGun_PDGid11_x960_Pt2.0To100.0_NTUP_92.root");
        }
        TDirectory *dir = (TDirectory *) f->Get(
                "/afs/cern.ch/user/j/jkiesele/eos_hgcal/FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_eta_2.3to2.5_timing_20170907/NTUP/partGun_PDGid11_x960_Pt2.0To100.0_NTUP_92.root:/ana");
        dir->GetObject("hgc", tree);

    }
    Init(tree);
}

HGCalSel::~HGCalSel() {
    if (!fChain) return;
    delete fChain->GetCurrentFile();
}

Int_t HGCalSel::GetEntry(Long64_t entry) {
// Read contents of entry.
    if (!fChain) return 0;
    return fChain->GetEntry(entry);
}

Long64_t HGCalSel::LoadTree(Long64_t entry) {
// Set the environment to read one entry
    if (!fChain) return -5;
    Long64_t centry = fChain->LoadTree(entry);
    if (centry < 0) return centry;
    if (fChain->GetTreeNumber() != fCurrent) {
        fCurrent = fChain->GetTreeNumber();
        Notify();
    }
    return centry;
}

void HGCalSel::Init(TTree *tree) {
    // The Init() function is called when the selector needs to initialize
    // a new tree or chain. Typically here the branch addresses and branch
    // pointers of the tree will be set.
    // It is normally not necessary to make changes to the generated
    // code, but the routine can be extended by the user if needed.
    // Init() will be called many times when running on PROOF
    // (once per file to be processed).

    // Set object pointer
    genpart_eta = 0;
    genpart_phi = 0;
    genpart_pt = 0;
    genpart_energy = 0;
    genpart_dvx = 0;
    genpart_dvy = 0;
    genpart_dvz = 0;
    genpart_ovx = 0;
    genpart_ovy = 0;
    genpart_ovz = 0;
    genpart_mother = 0;
    genpart_exphi = 0;
    genpart_exeta = 0;
    genpart_exx = 0;
    genpart_exy = 0;
    genpart_fbrem = 0;
    genpart_pid = 0;
    genpart_gen = 0;
    genpart_reachedEE = 0;
    genpart_fromBeamPipe = 0;
    genpart_posx = 0;
    genpart_posy = 0;
    genpart_posz = 0;
    rechit_eta = 0;
    rechit_phi = 0;
    rechit_pt = 0;
    rechit_energy = 0;
    rechit_x = 0;
    rechit_y = 0;
    rechit_z = 0;
    rechit_time = 0;
    rechit_thickness = 0;
    rechit_layer = 0;
    rechit_wafer = 0;
    rechit_cell = 0;
    rechit_detid = 0;
    rechit_isHalf = 0;
    rechit_flags = 0;
    rechit_cluster2d = 0;
    cluster2d_eta = 0;
    cluster2d_phi = 0;
    cluster2d_pt = 0;
    cluster2d_energy = 0;
    cluster2d_x = 0;
    cluster2d_y = 0;
    cluster2d_z = 0;
    cluster2d_layer = 0;
    cluster2d_nhitCore = 0;
    cluster2d_nhitAll = 0;
    cluster2d_multicluster = 0;
    cluster2d_rechits = 0;
    cluster2d_rechitSeed = 0;
    multiclus_eta = 0;
    multiclus_phi = 0;
    multiclus_pt = 0;
    multiclus_energy = 0;
    multiclus_z = 0;
    multiclus_slopeX = 0;
    multiclus_slopeY = 0;
    multiclus_cluster2d = 0;
    multiclus_cl2dSeed = 0;
    multiclus_firstLay = 0;
    multiclus_lastLay = 0;
    multiclus_NLay = 0;
    simcluster_eta = 0;
    simcluster_phi = 0;
    simcluster_pt = 0;
    simcluster_energy = 0;
    simcluster_simEnergy = 0;
    simcluster_hits = 0;
    simcluster_fractions = 0;
    simcluster_layers = 0;
    simcluster_wafers = 0;
    simcluster_cells = 0;
    pfcluster_eta = 0;
    pfcluster_phi = 0;
    pfcluster_pt = 0;
    pfcluster_energy = 0;
    pfcluster_correctedEnergy = 0;
    pfcluster_hits = 0;
    pfcluster_fractions = 0;
    calopart_eta = 0;
    calopart_phi = 0;
    calopart_pt = 0;
    calopart_energy = 0;
    calopart_simEnergy = 0;
    calopart_simClusterIndex = 0;
    track_eta = 0;
    track_phi = 0;
    track_pt = 0;
    track_energy = 0;
    track_charge = 0;
    track_posx = 0;
    track_posy = 0;
    track_posz = 0;
    // Set branch addresses and branch pointers
    if (!tree) return;
    fChain = tree;
    fCurrent = -1;
    fChain->SetMakeClass(1);

    fChain->SetBranchAddress("event", &event, &b_event);
    fChain->SetBranchAddress("lumi", &lumi, &b_lumi);
    fChain->SetBranchAddress("run", &run, &b_run);
    fChain->SetBranchAddress("vtx_x", &vtx_x, &b_vtx_x);
    fChain->SetBranchAddress("vtx_y", &vtx_y, &b_vtx_y);
    fChain->SetBranchAddress("vtx_z", &vtx_z, &b_vtx_z);
    fChain->SetBranchAddress("genpart_eta", &genpart_eta, &b_genpart_eta);
    fChain->SetBranchAddress("genpart_phi", &genpart_phi, &b_genpart_phi);
    fChain->SetBranchAddress("genpart_pt", &genpart_pt, &b_genpart_pt);
    fChain->SetBranchAddress("genpart_energy", &genpart_energy, &b_genpart_energy);
    fChain->SetBranchAddress("genpart_dvx", &genpart_dvx, &b_genpart_dvx);
    fChain->SetBranchAddress("genpart_dvy", &genpart_dvy, &b_genpart_dvy);
    fChain->SetBranchAddress("genpart_dvz", &genpart_dvz, &b_genpart_dvz);
    fChain->SetBranchAddress("genpart_ovx", &genpart_ovx, &b_genpart_ovx);
    fChain->SetBranchAddress("genpart_ovy", &genpart_ovy, &b_genpart_ovy);
    fChain->SetBranchAddress("genpart_ovz", &genpart_ovz, &b_genpart_ovz);
    fChain->SetBranchAddress("genpart_mother", &genpart_mother, &b_genpart_mother);
    fChain->SetBranchAddress("genpart_exphi", &genpart_exphi, &b_genpart_exphi);
    fChain->SetBranchAddress("genpart_exeta", &genpart_exeta, &b_genpart_exeta);
    fChain->SetBranchAddress("genpart_exx", &genpart_exx, &b_genpart_exx);
    fChain->SetBranchAddress("genpart_exy", &genpart_exy, &b_genpart_exy);
    fChain->SetBranchAddress("genpart_fbrem", &genpart_fbrem, &b_genpart_fbrem);
    fChain->SetBranchAddress("genpart_pid", &genpart_pid, &b_genpart_pid);
    fChain->SetBranchAddress("genpart_gen", &genpart_gen, &b_genpart_gen);
    fChain->SetBranchAddress("genpart_reachedEE", &genpart_reachedEE, &b_genpart_reachedEE);
    fChain->SetBranchAddress("genpart_fromBeamPipe", &genpart_fromBeamPipe, &b_genpart_fromBeamPipe);
    fChain->SetBranchAddress("genpart_posx", &genpart_posx, &b_genpart_posx);
    fChain->SetBranchAddress("genpart_posy", &genpart_posy, &b_genpart_posy);
    fChain->SetBranchAddress("genpart_posz", &genpart_posz, &b_genpart_posz);
    fChain->SetBranchAddress("rechit_eta", &rechit_eta, &b_rechit_eta);
    fChain->SetBranchAddress("rechit_phi", &rechit_phi, &b_rechit_phi);
    fChain->SetBranchAddress("rechit_pt", &rechit_pt, &b_rechit_pt);
    fChain->SetBranchAddress("rechit_energy", &rechit_energy, &b_rechit_energy);
    fChain->SetBranchAddress("rechit_x", &rechit_x, &b_rechit_x);
    fChain->SetBranchAddress("rechit_y", &rechit_y, &b_rechit_y);
    fChain->SetBranchAddress("rechit_z", &rechit_z, &b_rechit_z);
    fChain->SetBranchAddress("rechit_time", &rechit_time, &b_rechit_time);
    fChain->SetBranchAddress("rechit_thickness", &rechit_thickness, &b_rechit_thickness);
    fChain->SetBranchAddress("rechit_layer", &rechit_layer, &b_rechit_layer);
    fChain->SetBranchAddress("rechit_wafer", &rechit_wafer, &b_rechit_wafer);
    fChain->SetBranchAddress("rechit_cell", &rechit_cell, &b_rechit_cell);
    fChain->SetBranchAddress("rechit_detid", &rechit_detid, &b_rechit_detid);
    fChain->SetBranchAddress("rechit_isHalf", &rechit_isHalf, &b_rechit_isHalf);
    fChain->SetBranchAddress("rechit_flags", &rechit_flags, &b_rechit_flags);
    fChain->SetBranchAddress("rechit_cluster2d", &rechit_cluster2d, &b_rechit_cluster2d);
    fChain->SetBranchAddress("cluster2d_eta", &cluster2d_eta, &b_cluster2d_eta);
    fChain->SetBranchAddress("cluster2d_phi", &cluster2d_phi, &b_cluster2d_phi);
    fChain->SetBranchAddress("cluster2d_pt", &cluster2d_pt, &b_cluster2d_pt);
    fChain->SetBranchAddress("cluster2d_energy", &cluster2d_energy, &b_cluster2d_energy);
    fChain->SetBranchAddress("cluster2d_x", &cluster2d_x, &b_cluster2d_x);
    fChain->SetBranchAddress("cluster2d_y", &cluster2d_y, &b_cluster2d_y);
    fChain->SetBranchAddress("cluster2d_z", &cluster2d_z, &b_cluster2d_z);
    fChain->SetBranchAddress("cluster2d_layer", &cluster2d_layer, &b_cluster2d_layer);
    fChain->SetBranchAddress("cluster2d_nhitCore", &cluster2d_nhitCore, &b_cluster2d_nhitCore);
    fChain->SetBranchAddress("cluster2d_nhitAll", &cluster2d_nhitAll, &b_cluster2d_nhitAll);
    fChain->SetBranchAddress("cluster2d_multicluster", &cluster2d_multicluster, &b_cluster2d_multicluster);
    fChain->SetBranchAddress("cluster2d_rechits", &cluster2d_rechits, &b_cluster2d_rechits);
    fChain->SetBranchAddress("cluster2d_rechitSeed", &cluster2d_rechitSeed, &b_cluster2d_rechitSeed);
    fChain->SetBranchAddress("multiclus_eta", &multiclus_eta, &b_multiclus_eta);
    fChain->SetBranchAddress("multiclus_phi", &multiclus_phi, &b_multiclus_phi);
    fChain->SetBranchAddress("multiclus_pt", &multiclus_pt, &b_multiclus_pt);
    fChain->SetBranchAddress("multiclus_energy", &multiclus_energy, &b_multiclus_energy);
    fChain->SetBranchAddress("multiclus_z", &multiclus_z, &b_multiclus_z);
    fChain->SetBranchAddress("multiclus_slopeX", &multiclus_slopeX, &b_multiclus_slopeX);
    fChain->SetBranchAddress("multiclus_slopeY", &multiclus_slopeY, &b_multiclus_slopeY);
    fChain->SetBranchAddress("multiclus_cluster2d", &multiclus_cluster2d, &b_multiclus_cluster2d);
    fChain->SetBranchAddress("multiclus_cl2dSeed", &multiclus_cl2dSeed, &b_multiclus_cl2dSeed);
    fChain->SetBranchAddress("multiclus_firstLay", &multiclus_firstLay, &b_multiclus_firstLay);
    fChain->SetBranchAddress("multiclus_lastLay", &multiclus_lastLay, &b_multiclus_lastLay);
    fChain->SetBranchAddress("multiclus_NLay", &multiclus_NLay, &b_multiclus_NLay);
    fChain->SetBranchAddress("simcluster_eta", &simcluster_eta, &b_simcluster_eta);
    fChain->SetBranchAddress("simcluster_phi", &simcluster_phi, &b_simcluster_phi);
    fChain->SetBranchAddress("simcluster_pt", &simcluster_pt, &b_simcluster_pt);
    fChain->SetBranchAddress("simcluster_energy", &simcluster_energy, &b_simcluster_energy);
    fChain->SetBranchAddress("simcluster_simEnergy", &simcluster_simEnergy, &b_simcluster_simEnergy);
    fChain->SetBranchAddress("simcluster_hits", &simcluster_hits, &b_simcluster_hits);
    fChain->SetBranchAddress("simcluster_fractions", &simcluster_fractions, &b_simcluster_fractions);
    fChain->SetBranchAddress("simcluster_layers", &simcluster_layers, &b_simcluster_layers);
    fChain->SetBranchAddress("simcluster_wafers", &simcluster_wafers, &b_simcluster_wafers);
    fChain->SetBranchAddress("simcluster_cells", &simcluster_cells, &b_simcluster_cells);
    fChain->SetBranchAddress("pfcluster_eta", &pfcluster_eta, &b_pfcluster_eta);
    fChain->SetBranchAddress("pfcluster_phi", &pfcluster_phi, &b_pfcluster_phi);
    fChain->SetBranchAddress("pfcluster_pt", &pfcluster_pt, &b_pfcluster_pt);
    fChain->SetBranchAddress("pfcluster_energy", &pfcluster_energy, &b_pfcluster_energy);
    fChain->SetBranchAddress("pfcluster_correctedEnergy", &pfcluster_correctedEnergy, &b_pfcluster_correctedEnergy);
    fChain->SetBranchAddress("pfcluster_hits", &pfcluster_hits, &b_pfcluster_hits);
    fChain->SetBranchAddress("pfcluster_fractions", &pfcluster_fractions, &b_pfcluster_fractions);
    fChain->SetBranchAddress("calopart_eta", &calopart_eta, &b_calopart_eta);
    fChain->SetBranchAddress("calopart_phi", &calopart_phi, &b_calopart_phi);
    fChain->SetBranchAddress("calopart_pt", &calopart_pt, &b_calopart_pt);
    fChain->SetBranchAddress("calopart_energy", &calopart_energy, &b_calopart_energy);
    fChain->SetBranchAddress("calopart_simEnergy", &calopart_simEnergy, &b_calopart_simEnergy);
    fChain->SetBranchAddress("calopart_simClusterIndex", &calopart_simClusterIndex, &b_calopart_simClusterIndex);
    fChain->SetBranchAddress("track_eta", &track_eta, &b_track_eta);
    fChain->SetBranchAddress("track_phi", &track_phi, &b_track_phi);
    fChain->SetBranchAddress("track_pt", &track_pt, &b_track_pt);
    fChain->SetBranchAddress("track_energy", &track_energy, &b_track_energy);
    fChain->SetBranchAddress("track_charge", &track_charge, &b_track_charge);
    fChain->SetBranchAddress("track_posx", &track_posx, &b_track_posx);
    fChain->SetBranchAddress("track_posy", &track_posy, &b_track_posy);
    fChain->SetBranchAddress("track_posz", &track_posz, &b_track_posz);
    Notify();
}

Bool_t HGCalSel::Notify() {
    // The Notify() function is called when a new file is opened. This
    // can be either for a new TTree in a TChain or when when a new TTree
    // is started when using PROOF. It is normally not necessary to make changes
    // to the generated code, but the routine can be extended by the
    // user if needed. The return value is currently not used.

    return kTRUE;
}

void HGCalSel::Show(Long64_t entry) {
// Print contents of entry.
// If entry is not specified, print current entry
    if (!fChain) return;
    fChain->Show(entry);
}

Int_t HGCalSel::Cut(Long64_t entry) {
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
    return 1;
}

#endif // #ifdef HGCalSel_cxx
