// ArrayFire_OCT.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define NOMINMAX
#include <iostream>
#include <fstream>
#include <arrayfire.h>
#include <chrono>

using namespace af;

struct MZIProcessingParameters {
    double dSlope;
    int nCorrSmoothing;
    int nFinalCutLeft;
    int nFinalCutRight;
    int nInitialPeakLeft;
    int nInitialPeakRight;
    int nInitialPeakRound;
    int nPaddingFactor;
    int nRawLineLength;
    int nRawMaskLeft;
    int nRawMaskRight;
    int nRawMaskRound;
    int nRawSpectrumMax;

    char* s_afX;
    char* s_afZPX;
    char* s_afRefMZI;
    char* s_afFitLine;

    char* key_afX;
    char* key_afZPX;
    char* key_afRefMZI;
    char* key_afFitLine;
};
struct DispersionProcessingParameters {
    int nGSMultiplier;
    int nDispersionLeft;
    int nDispersionRight;
    int nDispersionRound;

    char* s_afDispersion;
    char* s_afRefDispersion;
    char* key_afDispersion;
    char* key_afRefDispersion;
};

int writeMZIParametersToFile(MZIProcessingParameters structToWrite) {
    std::ofstream outFile;
    outFile.open("MZIProcessingParameters.dat", std::ios::binary | std::ios::out);
    if (outFile.fail())
    {
        std::cout << "Failed to open stream to save MZIProcessingParameters.dat\n";
        return 1;
    }
    outFile.write((char*) &structToWrite, sizeof(structToWrite));
    outFile.close();
    
    return 0;
}
int writeDispersionParametersToFile(DispersionProcessingParameters structToWrite) {
    std::ofstream outFile;
    outFile.open("DispersionProcessingParameters.dat", std::ios::binary | std::ios::out);
    if (outFile.fail())
    {
        std::cout << "Failed to open stream to save DispersionProcessingParameters.dat\n";
        return 1;
    }
    outFile.write((char*)&structToWrite, sizeof(structToWrite));
    outFile.close();

    return 0;
}
int readMZIParameterFile(MZIProcessingParameters& structToReadInto) {
    std::ifstream inFile;
    inFile.open("MZIProcessingParameters.dat", std::ios::binary | std::ios::in);
    if (inFile.fail()) {
        std::cout << "Failed to open MZIProcessingParameters.dat\n";
        return 1;
    }
    inFile.read((char*)&structToReadInto, sizeof(structToReadInto));
    inFile.close();

    //you need to test that these can read and write structures to files
    return 0;
}
int readDispersionParameterFile(DispersionProcessingParameters& structToReadInto) {
    std::ifstream inFile;
    inFile.open("DispersionProcessingParameters.dat", std::ios::binary | std::ios::in);
    if (inFile.fail()) {
        std::cout << "Failed to open DispersionProcessingParameters.dat\n";
        return 1;
    }
    inFile.read((char*)&structToReadInto, sizeof(structToReadInto));
    inFile.close();

    return 0;
}

af::array mapPositionsToUniformInterval(const af::array& knownPositions, const af::array& desiredPositions) {

    af::array ret = desiredPositions;
    //distances to normalize between two data points
    af::array dists = diff1(knownPositions);
    
    //get indices of two nearest data points to desired position ... where / which index within the [nodePositions[0], nodePositions[1], nodePositions[2]] interval are the positions?
    af::array idxs = af::constant(0.0, 1, ret.dims(0), s32);
    
    //int nSegments = 4;
    //std::cout<< "\nkPos.dims(): " <<knownPositions.dims();
    //std::cout << "\ndesPos.dims(): " << desiredPositions.dims();
    //array temp0;
    //array temp1;
    //or (int ii = 0; ii < ret.dims(0); ii += ret.dims(0) / nSegments) {
    
    gfor(af::seq i, ret.dims(0))
        {
            af::array temp = af::tile(ret(i), knownPositions.dims(0));
            //temp0 = af::count(temp > knownPositions).as(s32) - 1;
            //temp1 = idxs(0, i);
            //std::cout << "\ntemp0.dims(): " << temp0.dims();
            //std::cout << "\ntemp1.dims(): " << temp1.dims() << std::endl;
                idxs(0, i) = af::count(temp > knownPositions).as(s32) - 1;
        }
    //    std::cout << "\nloop " << ii << "completed";
    //}
    idxs(idxs < 0) = 0;
    idxs(idxs > (knownPositions.dims(0) - 1)) = knownPositions.dims(0) - 1;

    //find offset values
    af::array minvals = knownPositions(idxs);

    //subtract offsets, normalize, and move to uniform index locations
    ret -= minvals;
    ret /= dists(idxs);
    ret += idxs.T();
    return ret;
}


int initializeMZIAlazar(af::array& afRawMZI, const int nRows, const int nColumns) {
    // hard coded parameters
    int nRawSpectrumMax = 16384 * 4;
    int nCorrSmoothing = 8; //moving-average window size
    float fCorrFactor = (float)1 / nCorrSmoothing;
    int nRawMaskLeft = 826;
    int nRawMaskRight = 6332;
    int nRawMaskRound = 32;
    int nPaddingFactor = 4;
    int nInitialPeakLeft = 73;
    int nInitialPeakRight = 860;
    int nInitialPeakRound = 16;
    int nFinalCutLeft = 1150;
    int nFinalCutRight = 6070;
    int nOffset;
    int nMidLength = afRawMZI.dims(0) / 2 + 1;
    int nMidPoint;
    int dMax, dMin;
    int nRange;
    int nLeft, nRight;
    double dSlope;

    // initial correlation
    af::array afMeanFilter = af::constant(fCorrFactor, nCorrSmoothing, 1, f32);
    array afCorrMZI = af::convolve1(afRawMZI, afMeanFilter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    array afRefMZI = af::mean(afCorrMZI, 1);
    array afRefFFT = af::fft(afRefMZI);
    array afFFT = af::fft(afCorrMZI);
    nOffset = std::round(0.5 * (afRawMZI.dims(0) + 1));
    
    //    this should have worked but something is broken with afShift. array.host<>() works on newly constructed arrays but not afShift.
    //    This is concerning b/c you'll need to use .host<>() to pass back out of an af::array
    afRefFFT = af::tile(afRefFFT, 1, afFFT.dims(1));
    array afCorr = af::shift(af::ifft(afRefFFT * af::conjg(afFFT)), afFFT.dims(0) / 2, 0);
    af::array afMaxValues, afMaxIndices;
    af::max(afMaxValues, afMaxIndices, afCorr, 0);
    afMaxIndices = afMaxIndices.as(s32);
    array afShift = afMaxIndices - nOffset;
    array afCorrelatedMZI(afRawMZI.dims(0), afRawMZI.dims(1));
    int* pn_afShift = afShift.host<int>();
    for (int ii = 0; ii < nColumns; ii++) {
        afCorrelatedMZI(span, ii) = af::shift(afRawMZI(span, ii), pn_afShift[ii]);
    }

    afRefMZI = af::mean(afCorrelatedMZI, 1);
    af::deviceGC();
    
    // perform initial mask
        // calculate initial mask
    array afMask = af::constant(0, nRows);
    afMask(af::seq(nRawMaskLeft, nRawMaskRight)) = 1;
    
        
        afMask(af::seq(nRawMaskLeft + 1, nRawMaskLeft + nRawMaskRound)) = 0.5 * (1 + cos((af::seq(nRawMaskRound - 1, 0, -1)) * (af::Pi / (nRawMaskRound - 1))));
        afMask(af::seq(nRawMaskRight - nRawMaskRound, nRawMaskRight - 1)) = 0.5 * (1 + cos((af::seq(0, nRawMaskRound - 1)) * (af::Pi / (nRawMaskRound - 1))));
        af::array afMaskMatrix = af::tile(afMask, 1, afRawMZI.dims(1));

         // apply initial mask
        int dMid = 0.5 * nRawSpectrumMax;

        af::array afMaskedMZI;
            //std::cout << "afRawMZI.dims(): " << afRawMZI.dims() << std::endl << "afCMZI.dims(): " << afCorrelatedMZI.dims() << std::endl << "afMaskedMZI.dims(): " << afMaskedMZI.dims() << "\n afMaskMatrix.dims(): " << afMaskMatrix.dims();
        //This is the line that I realized I had my line and frame dimensions swapped on afRawMZI, smh.
        afMaskedMZI = (afCorrelatedMZI - dMid) * afMaskMatrix;

        af::deviceGC();
    // create indexing arrays, zero pad, then initial profiles
        // Create indexing arrays
    af::array afX = af::seq(afRawMZI.dims(0));
    af::array afZPX = af::seq(afRawMZI.dims(0) * nPaddingFactor) / nPaddingFactor;
    array afIndex = af::tile(afZPX, 1, afRawMZI.dims(1));


        // zero pad MZI
    afFFT = fft(afMaskedMZI);
    afFFT(nMidLength, span) = afFFT(nMidLength, span) / 2.0;

    array afPaddedFFT = af::constant(0, nPaddingFactor * afRawMZI.dims(0), afRawMZI.dims(1));
    afPaddedFFT = afPaddedFFT.as(c64);
    
    afPaddedFFT(af::seq(0, nMidLength), span) = afFFT(af::seq(0, nMidLength), span); 
    afPaddedFFT(af::seq(af::end - nMidLength + 2, end), span) = afFFT(seq(end - nMidLength + 2, end), span);

    array afZPMZI = af::ifft(afPaddedFFT) * nPaddingFactor;
    afZPMZI = af::real(afZPMZI);
        // Calculate initial profiles
    array afProfileMZI = fft(afZPMZI);


    af::deviceGC();
    // cut out peak
        // Create mask to cut peak
    afMask = af::constant(1, afProfileMZI.dims(0));
    afMask(af::seq(nInitialPeakLeft), span) = 0;
    afMask(af::seq(nInitialPeakLeft + 1, nInitialPeakLeft + nInitialPeakRound), span) = 0.5 * (1 + af::cos( (af::seq(nInitialPeakRound - 1, 0, -1)) * (af::Pi / (nInitialPeakRound - 1))));

    afMask(af::seq(nInitialPeakRight - nInitialPeakRound, nInitialPeakRight - 1), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound)) * (af::Pi / (nInitialPeakRound - 1))));
    afMask(af::seq(nInitialPeakRight, end), span) = 0;

    afMask = af::tile(afMask, 1, afProfileMZI.dims(1));
        // Apply mask
    array afMZIPeak = afProfileMZI * afMask;

    af::deviceGC();
    // calculate spectrum from peak
    array afSpectrum = af::ifft(afMZIPeak);
    // Calculate abs and angle
        // ^this is the comment from Hyle but it only calculates angle
    af::array afAngle = af::arg(afSpectrum);
    
    // "Unwrap" phase jumps
    af::array afDP = af::diff1(afAngle, 0); 
    af::array afDP_corr = afDP / (2 * af::Pi);
        // floor elements that == 0.5
    af::array afHalf = constant(0.5f, afDP_corr.dims(0), afDP_corr.dims(1));
    af::array afIsHalf = afDP_corr == afHalf;
    afDP_corr(afIsHalf) = af::floor(afDP_corr(afIsHalf));
        // ceil elements that == -0.5
    afHalf = -afHalf;
    afIsHalf = afDP_corr == afHalf;
    afDP_corr(afIsHalf) = af::ceil(afDP_corr(afIsHalf));
        // round all other elements    
    afDP_corr = af::round(afDP_corr);
        // don't correct where afDP < pi
    af::array afIsJumped = af::abs(afDP) <= af::Pi;
    afDP_corr(afIsJumped) = 0;
    // apply corrections
    afAngle(seq(1, end), span) = afAngle(seq(1, end), span) - (2 * af::Pi) * af::scan(afDP_corr, 0, AF_BINARY_ADD, true);

    af::deviceGC();
    // clean ends
    af::array afZPX_left, afZPX_right, x_hat;
    afZPX_left = afZPX(seq((nRawMaskLeft + 1) * nPaddingFactor, (nRawMaskLeft + nRawMaskRound) * nPaddingFactor));
    afZPX_right = afZPX(seq((nRawMaskRight - nRawMaskRound) * nPaddingFactor, (nRawMaskRight - 1) * nPaddingFactor));
    afZPX = afZPX.as(f64);
    x_hat = x_hat.as(f64);
    afZPX_left = afZPX_left.as(f64);
    afZPX_right = afZPX_right.as(f64);

    for (int nLine = 0; nLine < afAngle.dims(1); nLine++) {
        // left
        x_hat = af::matmul(af::pinverse(afZPX_left), afAngle(seq((nRawMaskLeft + 1) * nPaddingFactor, (nRawMaskLeft + nRawMaskRound) * nPaddingFactor), nLine));
        afAngle(seq(0, nRawMaskLeft * nPaddingFactor), nLine) = af::matmul(afZPX(seq(0, nRawMaskLeft * nPaddingFactor)), x_hat);

        // right
        x_hat = af::matmul( af::pinverse(afZPX_right), afAngle(seq((nRawMaskRight - nRawMaskRound) * nPaddingFactor, (nRawMaskRight - 1) * nPaddingFactor), nLine) );
        afAngle(seq(nRawMaskRight * nPaddingFactor, end), nLine) = af::matmul(afZPX(seq(nRawMaskRight * nPaddingFactor, end)), x_hat);
    }

    // Get rid of 2pi ambiguity
    nMidPoint = std::round(0.5 * (nRawMaskLeft + nRawMaskRight) * nPaddingFactor);
    array af2pi = 2 * af::Pi * af::round(afAngle(nMidPoint, span) / (2 * af::Pi));
    array afCorrectedAngle = afAngle - tile(af2pi, afAngle.dims(0));
    afMaxValues = af::max(afCorrectedAngle(nMidPoint, span));
    array afMinValues = af::min(afCorrectedAngle(nMidPoint, span));

    if (afMaxValues.scalar<double>() - afMinValues.scalar<double>() > af::Pi) {
        afAngle = afAngle + af::Pi;
        af2pi = 2 * af::Pi * af::round(afAngle(nMidPoint, span) / (2 * af::Pi));
        afCorrectedAngle = afAngle - af::tile(af2pi, afAngle.dims(0)) - af::Pi;
    }

    // Perform fit from central 50% of average corrected angle
    af::array afCorrectedLine = af::mean(afCorrectedAngle, 1);
    nMidPoint = (nRawMaskRight + nRawMaskLeft) / 2;
    nRange = (nRawMaskRight - nRawMaskLeft) / 2;
    nLeft = std::round((nMidPoint - 0.5 * nRange) * nPaddingFactor);
    nRight = std::round((nMidPoint + 0.5 * nRange) * nPaddingFactor);

        // lines 156 and 157 of Init'MZIA'.m, ie polyfit and polysolve
    af::array A = af::constant(1, afZPX(seq(nLeft, nRight)).dims(0), 2, f64);
    A(span, 0) = afZPX(seq(nLeft, nRight)); // A should be in the format [ vec_ZPX, vec_ones ] now
    af::array B = afCorrectedLine(seq(nLeft, nRight));
    
    x_hat = x_hat.as(f32);
    A = A.as(f32);
    B = B.as(f32);
    af::deviceGC();  
    std::cout << "\nA.dims(): " << A.dims();
    std::cout << "\nB.dims(): " << B.dims();
    x_hat = af::matmul(af::pinverse(A), B); //solve for slope and y-intercept, m and b respectively
    std::cout << "\nx_hat.dims(): " << x_hat.dims();
    A = af::constant(1, afZPX.dims(0), 2, f64);
    A(span, 0) = afZPX(span);
    A = A.as(f32);
    af::array afFitLine = af::matmul(A, x_hat); // afFitLine should now hold the positions on the line...
    //      ... afFitLine is equivalent to the return of polyval ...
    //      ... a.k.a. B_hat

   
    float fSlope = x_hat(0).scalar<float>();
    dSlope = fSlope;
    
    // Check corrections for error
        // calculate error
    array afError = afCorrectedAngle - af::tile(afFitLine, 1, afCorrectedAngle.dims(1));
        // calculate new indexes
    afIndex = afIndex + afError / dSlope;
    
    af::deviceGC();
    // perform interpolation
    array afFinalMZI = af::constant(0, afX.dims(0), afZPMZI.dims(1));
    af::deviceGC();
    // lines 170 through 173 from Init'MZIA'.m
    // calc desired positions based on (possibly non-equidistant) node positions
    array mappedPositions_x, interpolatedArray, temp;
    afZPMZI = af::real(afZPMZI);
    afZPMZI = afZPMZI.as(f64);
        
    for (int ii = 0; ii < afIndex.dims(1); ii++) {
        mappedPositions_x = mapPositionsToUniformInterval(afIndex(span, ii), afX);// mPTUI(knownPositions, desiredPositions)
        //std::cout << mappedPositions_x.type() << " <-- mappedPositions_x.type() \n";
        //std::cout << afZPMZI.type() << " <-- afZPMZI.type() \n";

        temp = afZPMZI(span, ii);
        temp = af::approx1(temp, mappedPositions_x, AF_INTERP_LINEAR, 0.0f);
        //std::cout << "\n temp.dims(): " << temp.dims() << " | temp.type(): " << temp.type();
        //std::cout << "\n afFinalMZI.dims(): " << afFinalMZI.dims() << " | afFinalMZI.type(): " << afFinalMZI.type() << std::endl;
        afFinalMZI(seq(0, 8191), ii) = temp;
    }

    //std::cout << "\nafFinalMZI.type(): " << afFinalMZI.type();
    //std::cout << "\nafFinalMZI.dim(): " << afFinalMZI.dims();

     // calculate a new profile
    array afFinalProfile = af::fft(afFinalMZI(af::seq(nFinalCutLeft, nFinalCutRight), span));
    //std::cout << "\nafFinalProfile.type(): " << afFinalProfile.type();
    //std::cout << "\nafFinalProfile.dim(): " << afFinalProfile.dims();

    array afFinalProfileLine = 20 * af::log10(af::mean(af::abs(afFinalProfile), 1));

    //std::cout << "\nafFinalProfileLine.type(): " << afFinalProfileLine.type();
    //std::cout << "\nafFinalProfileLine.dim(): " << afFinalProfileLine.dims();

    // set params in structure to save to file
    MZIProcessingParameters MZIParams;
    MZIParams.dSlope = dSlope;
    MZIParams.nCorrSmoothing = nCorrSmoothing;
    MZIParams.nFinalCutLeft = nFinalCutLeft;
    MZIParams.nFinalCutRight = nFinalCutRight;
    MZIParams.nInitialPeakLeft = nInitialPeakLeft;
    MZIParams.nInitialPeakRight = nInitialPeakRight;
    MZIParams.nInitialPeakRound = nInitialPeakRound;
    MZIParams.nPaddingFactor = nPaddingFactor;
    MZIParams.nRawLineLength = afRawMZI.dims(0);
    MZIParams.nRawMaskLeft = nRawMaskLeft;
    MZIParams.nRawMaskRight = nRawMaskRight;
    MZIParams.nRawMaskRound = nRawMaskRound;
    MZIParams.nRawSpectrumMax = nRawSpectrumMax;

    char s_afX[] = "afX";
    char s_afZPX[] = "afZPX";
    char s_afRefMZI[] = "afRefMZI";
    char s_afFitLine[] = "afFitLine";
    char key_afX[] = "pg01";
    char key_afZPX[] = "pg02";
    char key_afRefMZI[] = "pg03";
    char key_afFitLine[] = "pg04";

    af::saveArray(key_afX, afX, s_afX, false);
    af::saveArray(key_afZPX, afZPX, s_afZPX, false);
    af::saveArray(key_afRefMZI, afRefMZI, s_afRefMZI, false);
    af::saveArray(key_afFitLine, afFitLine, s_afFitLine, false);

    MZIParams.s_afX = s_afX;
    MZIParams.s_afZPX = s_afZPX;
    MZIParams.s_afRefMZI = s_afRefMZI;
    MZIParams.s_afFitLine = s_afFitLine;
    MZIParams.key_afX = key_afX;
    MZIParams.key_afZPX = key_afZPX;
    MZIParams.key_afRefMZI = key_afRefMZI;
    MZIParams.key_afFitLine = key_afFitLine;
    writeMZIParametersToFile(MZIParams);
    

    return 0;
}

int initializeDispersionAlazar(af::array& afRawMZI, af::array& afRawOCT, int nRows, int nColumns) {
    MZIProcessingParameters MZIParams;
    readMZIParameterFile(MZIParams);

    // parameters from initializeMZIAlazar()
    int nRawSpectrumMax = MZIParams.nRawSpectrumMax;
    int nCorrSmoothing = MZIParams.nCorrSmoothing; //moving-average window size
    float fCorrFactor = (float)1 / nCorrSmoothing;
    int nRawMaskLeft = MZIParams.nRawMaskLeft;
    int nRawMaskRight = MZIParams.nRawMaskRight;
    int nRawMaskRound = MZIParams.nRawMaskRound;
    int nPaddingFactor = MZIParams.nPaddingFactor;
    int nInitialPeakLeft = MZIParams.nInitialPeakLeft;
    int nInitialPeakRight = MZIParams.nInitialPeakRight;
    int nInitialPeakRound = MZIParams.nInitialPeakRound;
    int nFinalCutLeft = MZIParams.nFinalCutLeft;
    int nFinalCutRight = MZIParams.nFinalCutRight;
    int nOffset = std::round(0.5 * (afRawMZI.dims(0) + 1));
    int nMidLength = afRawMZI.dims(0) / 2 + 1;
    int nMidPoint;
    int dMax, dMin;
    int nRange;
    int nLeft, nRight;
    double dSlope = MZIParams.dSlope;

    // arrays from initializeMZIAlazar()
    array afX = af::readArray("afX", "pg01");
    array afZPX = af::readArray("afZPX", "pg02");
    array afRefMZI = af::readArray("afRefMZI", "pg03");
    array afFitLine = af::readArray("afFitLine", "pg04");

    // reading from saved s_af and key_af wasn't working so I just hardcoded it for now. I thinkn the root of this issue is my char[]'s from MZIParams aren't reading in.
        //std::cout << "\nMZIPs.s_afX: " << MZIParams.s_afX << "\n" << MZIParams.key_afX;
        //af::array afX = af::readArray(MZIParams.s_afX, MZIParams.key_afX);
        //af::array afZPX = af::readArray(MZIParams.s_afZPX, MZIParams.key_afZPX);
        //af::array afRefMZI = af::readArray(MZIParams.s_afRefMZI, MZIParams.key_afRefMZI);
        //af::array afFitLine = af::readArray(MZIParams.s_afFitLine, MZIParams.key_afFitLine);
        //std::cout << "afFitLine.dims()|type(): " << afFitLine.dims() << "\n" << afFitLine.type();

    // hard coded parameters
    int nGSMultiplier = 4;
    int nDispersionLeft = 335;
    int nDispersionRight = 355;
    int nDispersionRound = 16;

    // initial correlation
    af::array afMeanFilter = af::constant(fCorrFactor, nCorrSmoothing, 1, f32);
    array afCorrMZI = af::convolve1(afRawMZI, afMeanFilter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
    array afRefDispersion = af::mean(afCorrMZI, 1);
    array afRefFFT = af::fft(afRefDispersion);
    array afFFT = af::fft(afCorrMZI);
    afRefFFT = af::tile(afRefFFT, 1, afFFT.dims(1));
    array afCorr = af::shift(af::ifft(afRefFFT * af::conjg(afFFT)), afFFT.dims(0) / 2, 0); // may need to tile afRefFFT out to [8192 100 1 1] to match afFFT for *()
    af::array afMaxValues, afMaxIndices;
    af::max(afMaxValues, afMaxIndices, afCorr, 0);
    array afShift = afMaxIndices - nOffset;

    
    afShift = afShift.as(s32);
    int* pnShift = afShift.host<int>();

    array afCorrelatedMZI(afRawMZI.dims(0), afRawMZI.dims(1), afRawMZI.type());   
    array afCorrelatedOCT(afRawOCT.dims(0), afRawOCT.dims(1), afRawOCT.type());
    //gfor(seq ii, 0, afRawMZI.dims(1)) {
    //    afCorrelatedMZI(span, ii) = af::shift(afRawMZI(span, ii), pnShift[ii]);
     //   afCorrelatedOCT(span, ii) = af::shift(afRawOCT(span, ii), pnShift[ii]);
    //}
    for (int ii = 0; ii < afRawMZI.dims(1); ii++) {
        afCorrelatedMZI(span, ii) = af::shift(afRawMZI(span, ii), pnShift[ii]);
        afCorrelatedOCT(span, ii) = af::shift(afRawOCT(span, ii), pnShift[ii]);
    }
    
    afRefDispersion = af::mean(afCorrelatedMZI, 1);

    af::deviceGC();
    // Perform initial mask
        // calculate initial mask
    array afMask = af::constant(0, nRows);
    afMask(af::seq(nRawMaskLeft, nRawMaskRight)) = 1;
    afMask(af::seq(nRawMaskLeft + 1, nRawMaskLeft + nRawMaskRound)) = 0.5 * (1 + cos((af::seq(nRawMaskRound - 1, 0, -1)) * (af::Pi / (nRawMaskRound - 1))));
    afMask(af::seq(nRawMaskRight - nRawMaskRound, nRawMaskRight - 1)) = 0.5 * (1 + cos((af::seq(0, nRawMaskRound - 1)) * (af::Pi / (nRawMaskRound - 1))));
    afMask(af::seq(nRawMaskRight, end)) = 0;
    af::array afMaskMatrix = af::tile(afMask, 1, afRawMZI.dims(1));
        // apply initial mask
    int dMid = 0.5 * nRawSpectrumMax;
    array afMaskedMZI = (afCorrelatedMZI - dMid) * afMaskMatrix;
    array afMaskedOCT = (afCorrelatedOCT - dMid) * afMaskMatrix;

    af::deviceGC();
    // create indexing arrays, zero pad, then initial profiles
        // Create indexing arrays
    afX = af::seq(afRawMZI.dims(0));
    afZPX = af::seq(afRawMZI.dims(0) * nPaddingFactor) / nPaddingFactor;
    array afIndex = af::tile(afZPX, 1, afRawMZI.dims(1));
        // zero pad MZI
    afFFT = fft(afMaskedMZI);
    afFFT(nMidLength, span) = afFFT(nMidLength, span) / 2.0;

    array afPaddedFFT = af::constant(0, nPaddingFactor * afRawMZI.dims(0), afRawMZI.dims(1));
    afPaddedFFT = afPaddedFFT.as(c64);

    afPaddedFFT(af::seq(0, nMidLength), span) = afFFT(af::seq(0, nMidLength), span);
    afPaddedFFT(af::seq(af::end - nMidLength + 2, end), span) = afFFT(seq(end - nMidLength + 2, end), span);

    array afZPMZI = af::ifft(afPaddedFFT) * nPaddingFactor;
    afZPMZI = af::real(afZPMZI);
        // zero pad OCT
    afFFT = fft(afMaskedOCT);
    afFFT(nMidLength, span) = afFFT(nMidLength, span) / 2.0;

    afPaddedFFT = af::constant(0, nPaddingFactor * afRawMZI.dims(0), afRawMZI.dims(1));
    afPaddedFFT = afPaddedFFT.as(c64);

    afPaddedFFT(af::seq(0, nMidLength), span) = afFFT(af::seq(0, nMidLength), span);
    afPaddedFFT(af::seq(af::end - nMidLength + 2, end), span) = afFFT(seq(end - nMidLength + 2, end), span);

    array afZPOCT = af::ifft(afPaddedFFT) * nPaddingFactor;
    afZPOCT = af::real(afZPOCT);
        // Calculate initial profiles
    array afProfileMZI = fft(afZPMZI);

    
    af::deviceGC();
    // cut out peak
        // Create mask to cut peak
    afMask = af::constant(1, afProfileMZI.dims(0));
    afMask(af::seq(nInitialPeakLeft), span) = 0;
    afMask(af::seq(nInitialPeakLeft + 1, nInitialPeakLeft + nInitialPeakRound), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound - 1, 0, -1)) * (af::Pi / (nInitialPeakRound - 1))));

    afMask(af::seq(nInitialPeakRight - nInitialPeakRound, nInitialPeakRight - 1), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound)) * (af::Pi / (nInitialPeakRound - 1))));
    afMask(af::seq(nInitialPeakRight, end), span) = 0;

    afMask = af::tile(afMask, 1, afProfileMZI.dims(1));
    // cut out peak
            // Create mask to cut peak
    afMask = af::constant(1, afProfileMZI.dims(0));
    afMask(af::seq(nInitialPeakLeft), span) = 0;
    afMask(af::seq(nInitialPeakLeft + 1, nInitialPeakLeft + nInitialPeakRound), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound - 1, 0, -1)) * (af::Pi / (nInitialPeakRound - 1))));

    afMask(af::seq(nInitialPeakRight - nInitialPeakRound, nInitialPeakRight - 1), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound)) * (af::Pi / (nInitialPeakRound - 1))));
    afMask(af::seq(nInitialPeakRight, end), span) = 0;

    afMask = af::tile(afMask, 1, afProfileMZI.dims(1));
        // Apply mask
    array afMZIPeak = afProfileMZI * afMask;

        
    af::deviceGC();
    // calculate spectrum from peak
    array afSpectrum = af::ifft(afMZIPeak);
    // Calculate abs and angle
        // ^this is the comment from Hyle but it only calculates angle
    af::array afAngle = af::arg(afSpectrum);

    // "Unwrap" phase jumps
    af::array afDP = af::diff1(afAngle, 0);
    af::array afDP_corr = afDP / (2 * af::Pi);
    // floor elements that == 0.5
    af::array afHalf = constant(0.5f, afDP_corr.dims(0), afDP_corr.dims(1));
    af::array afIsHalf = afDP_corr == afHalf;
    afDP_corr(afIsHalf) = af::floor(afDP_corr(afIsHalf));
    // ceil elements that == -0.5
    afHalf = -afHalf;
    afIsHalf = afDP_corr == afHalf;
    afDP_corr(afIsHalf) = af::ceil(afDP_corr(afIsHalf));
    // round all other elements    
    afDP_corr = af::round(afDP_corr);
    // don't correct where afDP < pi
    af::array afIsJumped = af::abs(afDP) <= af::Pi;
    afDP_corr(afIsJumped) = 0;
    // apply corrections
    afAngle(seq(1, end), span) = afAngle(seq(1, end), span) - (2 * af::Pi) * af::scan(afDP_corr, 0, AF_BINARY_ADD, true);

        
    af::deviceGC();
    // clean ends
    af::array afZPX_left, afZPX_right, x_hat;
    afZPX_left = afZPX(seq((nRawMaskLeft + 1) * nPaddingFactor, (nRawMaskLeft + nRawMaskRound) * nPaddingFactor));
    afZPX_right = afZPX(seq((nRawMaskRight - nRawMaskRound) * nPaddingFactor, (nRawMaskRight - 1) * nPaddingFactor));
    afZPX = afZPX.as(f64);
    x_hat = x_hat.as(f64);
    afZPX_left = afZPX_left.as(f64);
    afZPX_right = afZPX_right.as(f64);

    for (int nLine = 0; nLine < afAngle.dims(1); nLine++) {
        // left
        x_hat = af::matmul(af::pinverse(afZPX_left), afAngle(seq((nRawMaskLeft + 1) * nPaddingFactor, (nRawMaskLeft + nRawMaskRound) * nPaddingFactor), nLine));
        afAngle(seq(0, nRawMaskLeft * nPaddingFactor), nLine) = af::matmul(afZPX(seq(0, nRawMaskLeft * nPaddingFactor)), x_hat);

        // right
        x_hat = af::matmul(af::pinverse(afZPX_right), afAngle(seq((nRawMaskRight - nRawMaskRound) * nPaddingFactor, (nRawMaskRight - 1) * nPaddingFactor), nLine));
        afAngle(seq(nRawMaskRight * nPaddingFactor, end), nLine) = af::matmul(afZPX(seq(nRawMaskRight * nPaddingFactor, end)), x_hat);
    }

    // Get rid of 2pi ambiguity
    nMidPoint = std::round(0.5 * (nRawMaskLeft + nRawMaskRight) * nPaddingFactor);
    array af2pi = 2 * af::Pi * af::round(afAngle(nMidPoint, span) / (2 * af::Pi));
    array afCorrectedAngle = afAngle - tile(af2pi, afAngle.dims(0));
    afMaxValues = af::max(afCorrectedAngle(nMidPoint, span));
    array afMinValues = af::min(afCorrectedAngle(nMidPoint, span));

    if (afMaxValues.scalar<double>() - afMinValues.scalar<double>() > af::Pi) {
        afAngle = afAngle + af::Pi;
        af2pi = 2 * af::Pi * af::round(afAngle(nMidPoint, span) / (2 * af::Pi));
        afCorrectedAngle = afAngle - af::tile(af2pi, afAngle.dims(0)) - af::Pi;
    }
        
    // apply corrections
        // calculate error
    array afError = afCorrectedAngle - af::tile(afFitLine, 1, afCorrectedAngle.dims(1));
        // calculate new indexes
    afIndex = afIndex + afError / dSlope;
        
        
    af::deviceGC();
    // do interpolation
    array afFinalMZI = af::constant(0, afX.dims(0), afZPMZI.dims(1));
    array afFinalOCT = af::constant(0, afX.dims(0), afZPOCT.dims(1));

           
    //      calc desired positions based on (possibly non-equidistant) node positions
    array mappedPositions_x, temp;
    for (int ii = 0; ii < afIndex.dims(1); ii++) {
        mappedPositions_x = mapPositionsToUniformInterval(afIndex(span, ii), afX);// mPTUI(knownPositions, desiredPositions)

        temp = afZPMZI(span, ii);
        temp = af::approx1(temp, mappedPositions_x, AF_INTERP_LINEAR, 0.0f);
        afFinalMZI(seq(0, 8191), ii) = temp;

        temp = afZPOCT(span, ii);
        temp = af::approx1(temp, mappedPositions_x, AF_INTERP_LINEAR, 0.0f);
        afFinalOCT(seq(0, 8191), ii) = temp;
    }
        
    // Final cut
    afFinalOCT = afFinalOCT(seq(nFinalCutLeft, nFinalCutRight), span);
        
    af::deviceGC();
    // Apply Gaussian reshaping
    //    create mask
    nRows = afFinalOCT.dims(0);
    nMidPoint = 0.5 * nRows;
        
    array afX2(seq(0, nRows));
    afX2 = ((afX2 - nMidPoint) / nMidPoint);
    afX2 = af::pow(afX2, 2);
    afMask = af::exp(-nGSMultiplier * afX2);

    afMask = af::tile(
        afMask.dims(0) / af::sum(afMask), afMask.dims(0)
                    ) * afMask;
        
    afMask = afMask(seq(0, end - 1));

    afMask = af::tile(afMask, 1, afFinalOCT.dims(1));
    //    apply mask
    afFinalOCT = afFinalOCT * afMask;
    //    calculate depth profiles
    array afProfileOCT = af::fft(afFinalOCT);
                
    af::deviceGC();
    // cut out peak
    //    calculate dispersion mask
    afMask = af::constant(1, afProfileOCT.dims(0));

    afMask(af::seq(0, nDispersionLeft-1)) = 0;
    afMask(af::seq(nDispersionLeft - nDispersionRound, nDispersionLeft - 1)) = 0.5 * (1 + cos((af::seq(nDispersionRound - 1, 0, -1)) * (af::Pi / (nDispersionRound - 1))));
    afMask(af::seq(nDispersionRight + 1, nDispersionRight + nDispersionRound)) = 0.5 * (1 + cos((af::seq(0, nDispersionRound - 1))     * (af::Pi / (nDispersionRound - 1))));
    afMask(af::seq(nDispersionRight + nDispersionRound, end)) = 0;
    afMask = af::tile(afMask, 1, afRawMZI.dims(1));
    //    apply dispersion mask
    array afOCTPeak = afProfileOCT * afMask;
        
    af::deviceGC();
        
    // Calculate angle
    afAngle = af::arg(af::ifft(afOCTPeak));

    //    Unwrap phase jumps
    afDP = af::diff1(afAngle, 0);
    afDP_corr = afDP / (2 * af::Pi);
    //      floor elements that == 0.5
    afHalf = constant(0.5f, afDP_corr.dims(0), afDP_corr.dims(1));
    afIsHalf = afDP_corr == afHalf;
    afDP_corr(afIsHalf) = af::floor(afDP_corr(afIsHalf));
    //      ceil elements that == -0.5
    afHalf = -afHalf;
    afIsHalf = afDP_corr == afHalf;
    afDP_corr(afIsHalf) = af::ceil(afDP_corr(afIsHalf));
    //      round all other elements    
    afDP_corr = af::round(afDP_corr);
    //      don't correct where afDP < pi
    afIsJumped = af::abs(afDP) <= af::Pi;
    afDP_corr(afIsJumped) = 0;
    //      apply corrections
    afAngle(seq(1, end), span) = afAngle(seq(1, end), span) - (2 * af::Pi) * af::scan(afDP_corr, 0, AF_BINARY_ADD, true);

    af::deviceGC();
    // Calculate average angle
    array afLine = af::mean(afAngle, 1);
    //    read afLine as afMeanAngle
        
    // Perform linear regression
    array pdX = afLine.dims(0); 
    afX = pdX;
        
    int n25Point = std::round(0.25 * afLine.dims(0));
    int n75Point = std::round(0.75 * afLine.dims(0));
    array A = af::constant(1, afX(seq(n25Point, n75Point)).dims(0), 2);
    A(span, 0) = afX(seq(n25Point, n75Point));
    x_hat = x_hat.as(f32);

    x_hat = af::matmul(af::pinverse(A), afLine(seq(n25Point, n75Point)));
    A = af::constant(1, afX.dims(0), 2);
    A(span, 0) = afX;
        
    array afFit = af::matmul(A, x_hat);

    // Calculate dispersion correction
    array afDispersion = afLine - afFit;
    
    DispersionProcessingParameters dispersionParams;
    dispersionParams.nGSMultiplier      = nGSMultiplier;
    dispersionParams.nDispersionLeft    = nDispersionLeft;
    dispersionParams.nDispersionRight   = nDispersionRight;
    dispersionParams.nDispersionRound   = nDispersionRound;

    char s_afDispersion[]       = "afDispersion";
    char s_afRefDispersion[]    = "afRefDispersion";
    char key_afDispersion[]     = "pg11";
    char key_afRefDispersion[]  = "pg12";

    af::saveArray(key_afDispersion, afDispersion, s_afDispersion, false);
    af::saveArray(key_afRefDispersion, afRefDispersion, s_afRefDispersion, false);
    af::deviceGC();

    dispersionParams.s_afDispersion      = s_afDispersion;
    dispersionParams.s_afRefDispersion   = s_afRefDispersion;
    dispersionParams.key_afDispersion    = key_afDispersion;
    dispersionParams.key_afRefDispersion = key_afRefDispersion;
    
    writeDispersionParametersToFile(dispersionParams);
    return 0;
}

int processOCTData(af::array& afOCT, af::array& afRawMZI, af::array& afRawOCT, int nRows, int nColumns) {
    auto start = std::chrono::high_resolution_clock::now();

    // Read in parameters from initializeMZIAlazar() and initializeDispersionAlazar()
    MZIProcessingParameters MZIParams;
    DispersionProcessingParameters dispersionParams;
    
    readMZIParameterFile(MZIParams);
    readDispersionParameterFile(dispersionParams);
    
    array afX             = af::readArray("afX", "pg01");
    array afZPX           = af::readArray("afZPX", "pg02");
    array afRefMZI        = af::readArray("afRefMZI", "pg03");
    array afFitLine       = af::readArray("afFitLine", "pg04");
    array afDispersion    = af::readArray("afDispersion", "pg11");
    array afRefDispersion = af::readArray("afRefDispersion", "pg12");
    
    //set local params
    //  MZIParams.
    int nRawSpectrumMax     = MZIParams.nRawSpectrumMax;
    int nCorrSmoothing      = MZIParams.nCorrSmoothing; //moving-average window size
    float fCorrFactor       = (float)1 / nCorrSmoothing;
    int nRawMaskLeft        = MZIParams.nRawMaskLeft;
    int nRawMaskRight       = MZIParams.nRawMaskRight;
    int nRawMaskRound       = MZIParams.nRawMaskRound;
    int nPaddingFactor      = MZIParams.nPaddingFactor;
    int nInitialPeakLeft    = MZIParams.nInitialPeakLeft;
    int nInitialPeakRight   = MZIParams.nInitialPeakRight;
    int nInitialPeakRound   = MZIParams.nInitialPeakRound;
    int nFinalCutLeft       = MZIParams.nFinalCutLeft;
    int nFinalCutRight      = MZIParams.nFinalCutRight;
    int nOffset = std::round(0.5 * (afRawMZI.dims(0) + 1));
    int nMidLength = afRawMZI.dims(0) / 2 + 1;
    int nMidPoint;
    int dMax, dMin;
    int nRange;
    int nLeft, nRight;
    double dSlope = MZIParams.dSlope;
    //  dispersionParams.
    int nGSMultiplier    = dispersionParams.nGSMultiplier;
    int nDispersionLeft  = dispersionParams.nDispersionLeft;
    int nDispersionRight = dispersionParams.nDispersionRight;
    int nDispersionRound = dispersionParams.nDispersionRound;

    auto stop = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(stop - start).count();



    std::cout << std::endl << "parameter read/declaration: " << duration_ms;

    start = std::chrono::high_resolution_clock::now();
    //try {
        // initial correlation
        af::array afMeanFilter = af::constant(fCorrFactor, nCorrSmoothing, 1, f32);
        array afCorrMZI = af::convolve1(afRawMZI, afMeanFilter, AF_CONV_DEFAULT, AF_CONV_SPATIAL);

        auto start1 = std::chrono::high_resolution_clock::now();
        array afRefCurrent = af::mean(afCorrMZI, 1);
        auto stop1 = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop1 - start1).count();
        std::cout << std::endl << "calculate reference: " << duration_ms;

        array afRefFFT = af::fft(afRefCurrent);
        
        start1 = std::chrono::high_resolution_clock::now();
        array afFFT = af::fft(afCorrMZI);
        stop1 = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop1 - start1).count();
        std::cout << std::endl << "forward FFT: " << duration_ms;

        afRefFFT = af::tile(afRefFFT, 1, afFFT.dims(1));
        array afCorr = af::shift(af::ifft(afRefFFT * af::conjg(afFFT)), afFFT.dims(0) / 2, 0); // may need to tile afRefFFT out to [8192 100 1 1] to match afFFT for *()
        af::array afMaxValues, afMaxIndices;
        af::max(afMaxValues, afMaxIndices, afCorr, 0);
        array afShift = afMaxIndices - nOffset;


        afShift = afShift.as(s32);
        int* pnShift = afShift.host<int>();

        array afCorrelatedMZI(afRawMZI.dims(0), afRawMZI.dims(1), afRawMZI.type());
        array afCorrelatedOCT(afRawOCT.dims(0), afRawOCT.dims(1), afRawOCT.type());
        //gfor(seq ii, 0, afRawMZI.dims(1)) {
        //    afCorrelatedMZI(span, ii) = af::shift(afRawMZI(span, ii), pnShift[ii]);
         //   afCorrelatedOCT(span, ii) = af::shift(afRawOCT(span, ii), pnShift[ii]);
        //}
        for (int ii = 0; ii < afRawMZI.dims(1); ii++) {
            afCorrelatedMZI(span, ii) = af::shift(afRawMZI(span, ii), pnShift[ii]);
            afCorrelatedOCT(span, ii) = af::shift(afRawOCT(span, ii), pnShift[ii]);
        }

        afRefCurrent = af::mean(afCorrelatedMZI, 1);
        stop = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        std::cout << std::endl << "initial correlation " << duration_ms;

        start = std::chrono::high_resolution_clock::now();


        af::deviceGC();

        start1 = std::chrono::high_resolution_clock::now();
        // Perform initial mask
            // calculate initial mask
        array afMask = af::constant(0, nRows);
        afMask(af::seq(nRawMaskLeft, nRawMaskRight)) = 1;
        afMask(af::seq(nRawMaskLeft + 1, nRawMaskLeft + nRawMaskRound)) = 0.5 * (1 + cos((af::seq(nRawMaskRound - 1, 0, -1)) * (af::Pi / (nRawMaskRound - 1))));
        afMask(af::seq(nRawMaskRight - nRawMaskRound, nRawMaskRight - 1)) = 0.5 * (1 + cos((af::seq(0, nRawMaskRound - 1)) * (af::Pi / (nRawMaskRound - 1))));
        afMask(af::seq(nRawMaskRight, end)) = 0;
        af::array afMaskMatrix = af::tile(afMask, 1, afRawMZI.dims(1));

        stop1 = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop1 - start1).count();
        std::cout << std::endl << "calculate mask: " << duration_ms;

            // apply initial mask
        int dMid = 0.5 * nRawSpectrumMax;
        array afMaskedMZI = (afCorrelatedMZI - dMid) * afMaskMatrix;
        array afMaskedOCT = (afCorrelatedOCT - dMid) * afMaskMatrix;

        stop = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        std::cout << std::endl << "do initial mask in ms: " << duration_ms;

        start = std::chrono::high_resolution_clock::now();


        af::deviceGC();
        // create indexing arrays, zero pad, then initial profiles
            // Create indexing arrays
        array afIndex = af::tile(afZPX, 1, afRawMZI.dims(1));
            // zero pad MZI
        afFFT = fft(afMaskedMZI);
        afFFT(nMidLength, span) = afFFT(nMidLength, span) / 2.0;

        array afPaddedFFT = af::constant(0, nPaddingFactor * afRawMZI.dims(0), afRawMZI.dims(1));
        afPaddedFFT = afPaddedFFT.as(c64); //pgreg002: can drop these all down to 32bits

        afPaddedFFT(af::seq(0, nMidLength), span) = afFFT(af::seq(0, nMidLength), span);
        afPaddedFFT(af::seq(af::end - nMidLength + 2, end), span) = afFFT(seq(end - nMidLength + 2, end), span);

        array afZPMZI = af::ifft(afPaddedFFT) * nPaddingFactor;
        afZPMZI = af::real(afZPMZI);
            // zero pad OCT
        afFFT = fft(afMaskedOCT);
        afFFT(nMidLength, span) = afFFT(nMidLength, span) / 2.0;

        afPaddedFFT = af::constant(0, nPaddingFactor * afRawMZI.dims(0), afRawMZI.dims(1));
        afPaddedFFT = afPaddedFFT.as(c64);

        afPaddedFFT(af::seq(0, nMidLength), span) = afFFT(af::seq(0, nMidLength), span);
        afPaddedFFT(af::seq(af::end - nMidLength + 2, end), span) = afFFT(seq(end - nMidLength + 2, end), span);

        start1 = std::chrono::high_resolution_clock::now();
        array afZPOCT = af::ifft(afPaddedFFT) * nPaddingFactor;
        stop1 = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop1 - start1).count();
        std::cout << std::endl << "reverse FFT: " << duration_ms;

        afZPOCT = af::real(afZPOCT);
            // Calculate initial profiles
        array afProfileMZI = fft(afZPMZI);

        stop = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop - start).count();
        std::cout << std::endl << "create indexing arrays, zero pad, initial profiles: " << duration_ms;

        //start = std::chrono::high_resolution_clock::now();


        af::deviceGC();
        // cut out peak
            // Create mask to cut peak
        afMask = af::constant(1, afProfileMZI.dims(0));
        afMask(af::seq(nInitialPeakLeft), span) = 0;
        afMask(af::seq(nInitialPeakLeft + 1, nInitialPeakLeft + nInitialPeakRound), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound - 1, 0, -1)) * (af::Pi / (nInitialPeakRound - 1))));

        afMask(af::seq(nInitialPeakRight - nInitialPeakRound, nInitialPeakRight - 1), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound)) * (af::Pi / (nInitialPeakRound - 1))));
        afMask(af::seq(nInitialPeakRight, end), span) = 0;

        afMask = af::tile(afMask, 1, afProfileMZI.dims(1));
        // cut out peak
            // Create mask to cut peak
        afMask = af::constant(1, afProfileMZI.dims(0));
        afMask(af::seq(nInitialPeakLeft), span) = 0;
        afMask(af::seq(nInitialPeakLeft + 1, nInitialPeakLeft + nInitialPeakRound), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound - 1, 0, -1)) * (af::Pi / (nInitialPeakRound - 1))));

        afMask(af::seq(nInitialPeakRight - nInitialPeakRound, nInitialPeakRight - 1), span) = 0.5 * (1 + af::cos((af::seq(nInitialPeakRound)) * (af::Pi / (nInitialPeakRound - 1))));
        afMask(af::seq(nInitialPeakRight, end), span) = 0;

        afMask = af::tile(afMask, 1, afProfileMZI.dims(1));
            // Apply mask
        array afMZIPeak = afProfileMZI * afMask;

        af::deviceGC();
        //  calculate spectrum from peak
        array afSpectrum = af::ifft(afMZIPeak);
        // Calculate abs and angle
            // ^this is the comment from Hyle but it only calculates angle
        start1 = std::chrono::high_resolution_clock::now();
        af::array afAngle = af::arg(afSpectrum);
     


        // "Unwrap" phase jumps
        af::array afDP = af::diff1(afAngle, 0);
        af::array afDP_corr = afDP / (2 * af::Pi);
        // floor elements that == 0.5
        af::array afHalf = constant(0.5f, afDP_corr.dims(0), afDP_corr.dims(1));
        af::array afIsHalf = afDP_corr == afHalf;
        afDP_corr(afIsHalf) = af::floor(afDP_corr(afIsHalf));
        // ceil elements that == -0.5
        afHalf = -afHalf;
        afIsHalf = afDP_corr == afHalf;
        afDP_corr(afIsHalf) = af::ceil(afDP_corr(afIsHalf));
        // round all other elements    
        afDP_corr = af::round(afDP_corr);
        // don't correct where afDP < pi
        af::array afIsJumped = af::abs(afDP) <= af::Pi;
        afDP_corr(afIsJumped) = 0;
        // apply corrections
        afAngle(seq(1, end), span) = afAngle(seq(1, end), span) - (2 * af::Pi) * af::scan(afDP_corr, 0, AF_BINARY_ADD, true);
        stop1 = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration<double, std::milli>(stop1 - start1).count();
        std::cout << std::endl << "calculate phase: " << duration_ms;

        af::deviceGC();
        //  clean ends
        af::array afZPX_left, afZPX_right, x_hat;
        afZPX_left = afZPX(seq((nRawMaskLeft + 1) * nPaddingFactor, (nRawMaskLeft + nRawMaskRound) * nPaddingFactor));
        afZPX_right = afZPX(seq((nRawMaskRight - nRawMaskRound) * nPaddingFactor, (nRawMaskRight - 1) * nPaddingFactor));
        afZPX = afZPX.as(f64);
        x_hat = x_hat.as(f64);
        afZPX_left = afZPX_left.as(f64);
        afZPX_right = afZPX_right.as(f64);

        for (int nLine = 0; nLine < afAngle.dims(1); nLine++) {
            // left
            x_hat = af::matmul(af::pinverse(afZPX_left), afAngle(seq((nRawMaskLeft + 1) * nPaddingFactor, (nRawMaskLeft + nRawMaskRound) * nPaddingFactor), nLine));
            afAngle(seq(0, nRawMaskLeft * nPaddingFactor), nLine) = af::matmul(afZPX(seq(0, nRawMaskLeft * nPaddingFactor)), x_hat);

            // right
            x_hat = af::matmul(af::pinverse(afZPX_right), afAngle(seq((nRawMaskRight - nRawMaskRound) * nPaddingFactor, (nRawMaskRight - 1) * nPaddingFactor), nLine));
            afAngle(seq(nRawMaskRight * nPaddingFactor, end), nLine) = af::matmul(afZPX(seq(nRawMaskRight * nPaddingFactor, end)), x_hat);
        }

        // Get rid of 2pi ambiguity
        nMidPoint = std::round(0.5 * (nRawMaskLeft + nRawMaskRight) * nPaddingFactor);
        array af2pi = 2 * af::Pi * af::round(afAngle(nMidPoint, span) / (2 * af::Pi));
        array afCorrectedAngle = afAngle - tile(af2pi, afAngle.dims(0));
        afMaxValues = af::max(afCorrectedAngle(nMidPoint, span));
        array afMinValues = af::min(afCorrectedAngle(nMidPoint, span));

        if (afMaxValues.scalar<double>() - afMinValues.scalar<double>() > af::Pi) {
            afAngle = afAngle + af::Pi;
            af2pi = 2 * af::Pi * af::round(afAngle(nMidPoint, span) / (2 * af::Pi));
            afCorrectedAngle = afAngle - af::tile(af2pi, afAngle.dims(0)) - af::Pi;
        }

        // Apply corrections
            // calculate error
        array afError = afCorrectedAngle - af::tile(afFitLine, 1, afCorrectedAngle.dims(1));
        // calculate new indexes
        afIndex = afIndex + afError / dSlope;

        af::deviceGC();
        // perform interpolation
        array afFinalMZI = af::constant(0, afX.dims(0), afZPMZI.dims(1));
        array afFinalOCT = af::constant(0, afX.dims(0), afZPOCT.dims(1));

        //  calc desired positions based on (possibly non-equidistant) node positions
        array mappedPositions_x, temp;
        for (int ii = 0; ii < afIndex.dims(1); ii++) {
            mappedPositions_x = mapPositionsToUniformInterval(afIndex(span, ii), afX);// mPTUI(knownPositions, desiredPositions)

            temp = afZPMZI(span, ii);
            temp = af::approx1(temp, mappedPositions_x, AF_INTERP_LINEAR, 0.0f);
            afFinalMZI(seq(0, 8191), ii) = temp;

            temp = afZPOCT(span, ii);
            temp = af::approx1(temp, mappedPositions_x, AF_INTERP_LINEAR, 0.0f);
            afFinalOCT(seq(0, 8191), ii) = temp;
        }

        // Final cut
        afFinalOCT = afFinalOCT(seq(nFinalCutLeft, nFinalCutRight), span);

        af::deviceGC();
        // Apply Gaussian reshaping
        //    create mask
        nRows = afFinalOCT.dims(0);
        nMidPoint = 0.5 * nRows;

        array afX2(seq(0, nRows));
        afX2 = ((afX2 - nMidPoint) / nMidPoint);
        afX2 = af::pow(afX2, 2);
        afMask = af::exp(-nGSMultiplier * afX2);

        afMask = af::tile(
            afMask.dims(0) / af::sum(afMask), afMask.dims(0)
        ) * afMask;

        afMask = afMask(seq(0, end - 1));

        afMask = af::tile(afMask, 1, afFinalOCT.dims(1));
        
        //    apply mask
        afFinalOCT = afFinalOCT * afMask;
        // Apply dispersion correction
        cdouble i_cdouble = { 0, 1 };
        afDispersion = af::exp(i_cdouble * afDispersion);
        afDispersion = af::tile(afDispersion, 1, afFinalOCT.dims(1));
        afOCT = afFinalOCT * afDispersion;
        //afOCT = afOCT.as(f32);
        af:deviceGC();
        
        /* */
    //}
    //catch (af::exception& e) {
    //    std::cout << e.what();
    //}
    
    return 0;
}

int runInitializeMZIAlazar(float* pfRawMZI, int nLineLength, int nFrameLength)
{
    int n;

    af::array afRawMZI(nLineLength, nFrameLength, pfRawMZI);
    n = initializeMZIAlazar(afRawMZI, nLineLength, nFrameLength);

    return n;
}

int runInitializeDispersionAlazar(float* pfRawMZI, float* pfRawOCT, int nLineLength, int nFrameLength) {
    int n;
    af::array afRawMZI(nLineLength, nFrameLength, pfRawMZI);
    af::array afRawOCT(nLineLength, nFrameLength, pfRawOCT);
    n = initializeDispersionAlazar(afRawMZI, afRawOCT, nLineLength, nFrameLength);

    return n;
}

extern "C" {
    __declspec(dllexport) int runProcessOCTDataAlazar(float* pfOCT, float* pfRawMZI, float* pfRawOCT, int nLineLength, int nFrameLength) {
    int n;
    af::array afRawMZI(nLineLength, nFrameLength, pfRawMZI);
    af::array afRawOCT(nLineLength, nFrameLength, pfRawOCT);
    af::array afOCT;
    n = processOCTData(afOCT, afRawMZI, afRawOCT, nLineLength, nFrameLength);
    pfOCT = afOCT.host<float>();

    return n;
}
}	// extern "C"

int testParamIO(af::array afRawMZI) {
    MZIProcessingParameters MZIParams;
    MZIParams.dSlope = 1.1;
    char s_afX[] = "afX";
    char key_afX[] = "pg01";
    MZIParams.s_afX = s_afX;
    MZIParams.key_afX = key_afX;

    af::array afX = seq(0, afRawMZI.dims(0));
    af_print( afX(seq(0,5)) );
    
    writeMZIParametersToFile(MZIParams);
    af::saveArray(MZIParams.key_afX, afX, MZIParams.s_afX, false);
    af_print(afX(seq(0, 5)));
    
    afX = constant(0, afRawMZI.dims(0));
    af_print(afX(seq(0, 5)));

    afX = af::readArray(MZIParams.s_afX, MZIParams.key_afX);
    af_print(afX(seq(0, 5)));
    
    return 0;
}

/*  int testPerformance(float* pfOCT, constexpr int nLines, const int nFrames) {
    int n = 0;

    float pfarray[nLines, nFrames];
    std::cout << pfOCT[0] << '\n';
    pfOCT[0] = 1;
    std::cout << pfOCT[0] << '\n';


    return n;
}
*/

int main()
{
    std::cout << "Hello World!\n";
    int nFrameLength = 100;
    int nLineLength = 8192;
    array afOCT;

    af::array afRawMZI = af::constant(0, nLineLength, nFrameLength, f32);
    afRawMZI = randu(afRawMZI.dims());

    af::array afRawOCT = af::constant(0, nLineLength, nFrameLength, u16);
    afRawOCT = randu(afRawOCT.dims());
    
    int n;
    //n = initializeMZIAlazar(afRawMZI, nLineLength, nFrameLength);
    //n = initializeDispersionAlazar(afRawMZI, afRawOCT, nLineLength, nFrameLength);
    
    auto start = std::chrono::high_resolution_clock::now();
        n = processOCTData(afOCT, afRawMZI, afRawOCT, nLineLength, nFrameLength);
    auto stop = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(stop - start).count();

    std::cout << std::endl << "duration in ms was: " << duration_ms;
    

    std::cout << "\nfunction returned: " << n << "\n";
    //pgreg002: now just pull the values off the GPU object afOCT and into a float[]
    //          then pass that float[] pointer to whatever function is calling it.

    

    return 0;
}