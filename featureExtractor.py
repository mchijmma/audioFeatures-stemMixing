# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 14:04:51 2017

@author: marco

"""
from __future__ import division
from datetime import datetime


import numpy as np 


import matplotlib.pyplot as plt
plt.style.use('ggplot')
import essentia
import librosa
import essentia.standard 
import essentia.streaming 
from essentia.standard import *
from essentia.streaming import MonoLoader, AudioLoader




from collections import OrderedDict
import os
import gc
import sys

#%%
    
# Definition of functions. 
    
# returns list with name of tracks that contain gInstrument
def getInfoTracks(type = 'All'):
  
  semitones = np.concatenate((np.linspace(-4,-0.5,8),np.linspace(0.5,4,8)))
  
  name_tracks = []
  raw_path = OrderedDict()
  stem_path = OrderedDict()
  stem_stereo_path = OrderedDict()
  entries = gPool.descriptorNames()
  for entry in entries:
      split = entry.split('.')
      if kInstrument == split[2]:
          name_tracks.append(split[1])
          
  name_tracks = list(set(name_tracks))
  
  if type == 'original':
      for name in name_tracks:
        
        
          raw_path[name] = gPool['track.%s.%s.raw_path' % (name, kInstrument)]
          stem_path[name] = gPool['track.%s.%s.stem_path_mono' % (name, kInstrument)]      
          stem_stereo_path[name] = gPool['track.%s.%s.stem_path_stereo' % (name, kInstrument)]
          
  else:
    
      for name in name_tracks:           
          
            raw_path[name] = gPool['track.%s.%s.raw_path' % (name, kInstrument)]
            stem_path[name] = gPool['track.%s.%s.stem_path_mono' % (name, kInstrument)]      
            stem_stereo_path[name] = gPool['track.%s.%s.stem_path_stereo' % (name, kInstrument)]
            
            for st in semitones:
              
                rname = raw_path[name]
                sname = stem_path[name]
                ssname = stem_stereo_path[name]
                
                name2 = name + '%+d' % (int(100*st))
                #raw
                name3 = rname.split('/')[-1].split('_')
                name3[1] = name3[1] + '%+d' % (int(100*st))
                name3 = '_'.join(name3)
                name4 = rname.split('/')
                name4[-1] = name3
                name3 = '/'.join(name4)
                raw_path[name2] = name3
                
                #stem
                name3 = sname.split('/')[-1].split('_')
                name3[1] = name3[1] + '%+d' % (int(100*st))
                name3 = '_'.join(name3)
                name4 = sname.split('/')
                name4[-1] = name3
                name3 = '/'.join(name4)
                stem_path[name2] = name3 
                
                #stem_stereo
                name3 = sname.split('/')[-1].split('_')
                name3[1] = name3[1] + '%+d' % (int(100*st))
                name3 = '_'.join(name3)
                name4 = ssname.split('/')
                name4[-1] = name3
                name3 = '/'.join(name4)
                stem_stereo_path[name2] = name3
                
    
      
  return name_tracks, raw_path, stem_path, stem_stereo_path 
   
#gNameTracks, gRawPath, gStemPath, gStemStereoPath = getInfoTracks()

# Load audio tracks into dicts
def load_audio(type = 'mono'):
    
    raw_audio = OrderedDict()
    stem_audio = OrderedDict()

    
    if 'mono' in type:
      # loads raw audio
      loader = MonoLoader()
      for name in gNameTracks:
          path = gRawPath[name]
          loader.configure(filename = path)
          pool = essentia.Pool()
          loader.audio >> (pool,'loader.audio')
          essentia.run(loader)
          
          print 'Raw track contains %d samples of Audio' % len(pool['loader.audio'])
          
          raw_audio[name] = pool['loader.audio']
          
          essentia.reset(loader)
  
        # loads stem audio
      for name in gNameTracks:
          path = gStemPath[name]
          loader.configure(filename = path)
          pool = essentia.Pool()
          loader.audio >> (pool,'loader.audio')
          essentia.run(loader)
          
          print 'Stem track contains %d samples of Audio' % len(pool['loader.audio'])
          
          stem_audio[name] = pool['loader.audio']
          
          essentia.reset(loader)
          
    elif 'stereo' in type:
    
      # loads raw audio Stereo:
      for name in gNameTracks:
          path = gRawPath[name]
          loader = AudioLoader(filename = path)
          pool = essentia.Pool()
          loader.audio >> (pool,'loader.audio')
          loader.sampleRate >> None
          loader.numberChannels >> None
          loader.md5 >> None
          loader.bit_rate >> None
          loader.codec >> None
          essentia.run(loader)
          
          print 'Raw Stereo track contains %d samples of Audio' % len(pool['loader.audio'])
          
          raw_audio[name] = pool['loader.audio']
          
          essentia.reset(loader)
    

    
      # loads stem stereo
      for name in gNameTracks:
          path = gStemStereoPath[name]
          loader = AudioLoader(filename = path)
          pool = essentia.Pool()
          loader.audio >> (pool,'loader.audio')
          loader.sampleRate >> None
          loader.numberChannels >> None
          loader.md5 >> None
          loader.bit_rate >> None
          loader.codec >> None
          essentia.run(loader)
          
          print 'Stem Stereo track contains %d samples of Audio' % len(pool['loader.audio'])
          
          stem_audio[name] = pool['loader.audio']
          
          essentia.reset(loader)
    
    return raw_audio, stem_audio


#gRawAudio, gStemAudio = load_audio(type = 'stereo')

# Plots stem and raw track given a name.
def plotStemRawTrack(name):
  
    plt.close()
    ax = plt.subplot(111)
    plt.rcParams['figure.figsize'] = (18,15)
    Time1=np.linspace(0, len(gStemAudio[name])/kSampleRate, num=len(gStemAudio[name]))
    
    
    lines1, = ax.plot(Time1, gStemAudio[name],'k', label='stem', alpha=1)
    lines2, = ax.plot(Time1, gRawAudio[name], 'c', label='raw',alpha=0.5)
    
    #Sets legend outside plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(handles=[lines1, lines2],loc='center', bbox_to_anchor=(0.8, -0.1),
          fancybox=True, shadow=True, ncol=2)

    plt.xlabel('time (s)')
    plt.ylabel('amplitude')     
    
# Plot spectrum
def plotStemRawSpectrum(rD, sD, type = 'spectrum', hop_size = 1024,
                        power = True, yaxis = 'log', cmap = 'summer',
                        colorbar = False):
  plt.close()

  if 'spectrum' in type:
    rD = librosa.logamplitude(np.abs(rD)**2, ref_power=np.nanmax)
    sD = librosa.logamplitude(np.abs(sD)**2, ref_power=np.nanmax)
  
  if power:
      rD = rD**2
      sD = sD**2
  
  
  
  plt.rcParams['figure.figsize'] = (18,15) 
  ax1 = plt.subplot(2,1,1)
  ax1.set_title('Raw')
  
  librosa.display.specshow(rD,
                           cmap=cmap, sr = kSampleRate, hop_length = hop_size,
                           y_axis=yaxis, x_axis='time')
                            
  if colorbar:
    plt.colorbar()
    lim = np.nanmax(np.abs(rD))
    plt.clim(-lim,lim)
    
  ax2 = plt.subplot(2,1,2)
  ax2.set_title('Stem')
  librosa.display.specshow(sD,
                           cmap=cmap, sr = kSampleRate, hop_length = hop_size,
                           y_axis=yaxis, x_axis='time')
  
  if colorbar:
    
    lim = np.nanmax(np.abs(sD))
    plt.clim(-lim,lim)
    plt.colorbar()

# Plots stem and raw track feature given a name.
def plotFeature(feature, name, hop_size = 1024):
  
    plt.close()
    ax = plt.subplot(111)
    plt.rcParams['figure.figsize'] = (18,15)
    Time1=np.linspace(0, len(gRawFeatures[name][feature])*hop_size/kSampleRate, num=len(gRawFeatures[name][feature]))
    
    
    lines1, = ax.plot(Time1, gStemFeatures[name][feature],'k', label='stem', alpha=0.7)
    lines2, = ax.plot(Time1, gRawFeatures[name][feature], 'c', label='raw',alpha=1)
    
    #Sets legend outside plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(handles=[lines1, lines2],loc='center', bbox_to_anchor=(0.8, -0.1),
          fancybox=True, shadow=True, ncol=2)
    plt.title(name + ': ' + feature)
    plt.xlabel('time (s)')
    plt.ylabel('feature value')     


# Gets a dict of features among all track, for bar plots.
def getDictFeature(_features, _feature):
  
  dictFeature = OrderedDict()

  for name in gNameTracks:
  
      for ft in np.asarray([_features[name][_feature]]):
        
          dictFeature[name] = ft[0]
          
  return dictFeature

def plotBarFeatures(_feature):
  plt.close
  gRawDC = getDictFeature(gRawFeatures, _feature)
  gStemDC = getDictFeature(gStemFeatures, _feature)
  plt.bar(range(len(gRawDC)), gRawDC.values(), align='center', color='cyan')
  plt.bar(range(len(gStemDC)), gStemDC.values(), align='center', color='black', alpha=0.4)
  plt.xticks(range(len(gRawDC)), gRawDC.keys())
  plt.title(_feature)
  plt.show()



#_audio = gStemStereoAudio[_name]

# Calculates cochleagram difference of audio track.
def getCochleagramDifference(_audio):
  

  w_l = Windowing(type = 'hann')
  gfcc_l = GFCC(inputSize = 2049, lowFrequencyBound = 30,
              numberBands = 70)
  stereoDemuxer = StereoDemuxer()
  powerSpectrum_l = PowerSpectrum()
  
  w_r = Windowing(type = 'hann')
  gfcc_r = GFCC(inputSize = 2049, lowFrequencyBound = 30,
              numberBands = 70)
  stereoDemuxer = StereoDemuxer()
  powerSpectrum_r = PowerSpectrum()
  
  pool = essentia.Pool()
  
  left, right = stereoDemuxer(_audio)  
    
  if not np.any(right):
    right = left

  frame_l = FrameGenerator(left, frameSize=kN, hopSize=kN//2)     
  frame_r = FrameGenerator(right, frameSize=kN, hopSize=kN//2)   
  
  for _frame_l, _frame_r in zip(frame_l, frame_r):
    
      a_l = powerSpectrum_l(w_l(_frame_l))
      bands_l, coeffs_l = gfcc_l(a_l)   

      a_r = powerSpectrum_r(w_r(_frame_r))
      bands_r, coeffs_r = gfcc_r(a_r)        
      #pool.add('panning.left_gfcc', coeffs) 
      pool.add('panning.CD', bands_r - bands_l) 
       
      
  return pool
  

# Calculates stereo panning spectrum
#_audio = gStemStereoAudio[_name]

def getStereoPanningSpectrum(_audio):
  
  w_l = Windowing(type = 'hann')
  stereoDemuxer = StereoDemuxer()
  spectrum_l = FFT(size = kN)
  
  w_r = Windowing(type = 'hann')
  stereoDemuxer = StereoDemuxer()
  spectrum_r = FFT(size = kN)
  
  pool = essentia.Pool()
  
  rms = RMS()

  freq_1 = int(np.round((250*kN+2)/kSampleRate))
  freq_2 = int(np.round((2500*kN+2)/kSampleRate))
  
  left, right = stereoDemuxer(_audio)  
    
  if not np.any(right):
    right = left

  frame_l = FrameGenerator(left, frameSize=kN, hopSize=kN//2)     
  frame_r = FrameGenerator(right, frameSize=kN, hopSize=kN//2)   
  
  for _frame_l, _frame_r in zip(frame_l, frame_r):
    
      # Calculates Stereo Panning Spectrum
    
      l = spectrum_l(w_l(_frame_l)) 
      r = spectrum_r(w_r(_frame_r))
      
      phi_l = np.abs( l*np.conj(r)) / (np.abs(l)**2)

      phi_r = np.abs( r*np.conj(l)) / (np.abs(r)**2)

      phi = 2 * np.abs(l*np.conj(r)) / (np.abs(l)**2 + np.abs(r)**2)

      delta = phi_l - phi_r
      
      delta_ = []
      
      for bin in delta:

          if bin > 0:
              delta_.append(1)
          elif bin < 0:
              delta_.append(-1)
          else:
              delta_.append(0)
      
      SPS = ( 1 - phi) * delta_  
      SPS = essentia.array(SPS)
      pool.add('panning.SPS', SPS) 
      
      P_total = rms(SPS)
      P_low = rms(SPS[0:freq_1])
      P_medium = rms(SPS[freq_1:freq_2])
      P_high = rms(SPS[freq_2::]) 
      
      
      pool.add('panning.P_total', P_total) 
      pool.add('panning.P_low', P_low )
      pool.add('panning.P_medium', P_medium )
      pool.add('panning.P_high', P_high )
      
      #Calculates Stereo Phase Spread:
      
      frequencies = np.linspace(1,(kN/2)+1, (kN/2)+1) * (kSampleRate)/(kN + 2)  
      
      erb = erbScale(30, 11025, 40)
      
      phase_l = np.angle(l)
      phase_r = np.angle(r)
      mag_l = np.abs(l)
      mag_r = np.abs(r)
      pool2 = essentia.Pool()
      
      for erb_f0 in erb:
          
          freqs = np.asarray([])
          
          for f in frequencies:
          
              if find_nearest(erb, f) == erb_f0:
                  freqs = np.append(freqs, f)
              elif freqs.size != 0:
                  break

              
                  
                 
          freq1 = int(np.round((freqs[0]*kN+2)/kSampleRate))
          freq2 = int(np.round((freqs[-1]*kN+2)/kSampleRate)) 
          
          if freq2 == kN/2:
              freq2 = freq2 + 1

      
          S_l = np.cos(2 * np.pi * (freqs / kSampleRate) + phase_l[freq1-1:freq2])
          S_r = np.cos(2 * np.pi * (freqs / kSampleRate) + phase_r[freq1-1:freq2])
          
          a_weight = np.mean(mag_l[freq1-1:freq2] + mag_r[freq1-1:freq2])
      
          delta_lr = a_weight * np.std(S_l - S_r) / np.std(S_l + S_r)
          
          if freq2 - freq1 == 0:
            
            #delta_lr = a_weight * np.mean(S_l - S_r) / np.mean(S_l + S_r)
            delta_lr = 0
          
          pool2.add('a', delta_lr)
        
          
      pool.add('panning.SSPS', pool2['a'])
  
      
  
  return pool


# Returns erbscale, this is for Stereo Phase Spread
def erbScale(lowFreq, highFreq, N):
  earQ = 9.26449
  minBW = 24.7
  #order = 1  
  #returns erb centre frequencies and bandwidth
  erb = (-(earQ*minBW) + np.exp(np.arange(1,N+1).T*(-np.log(highFreq + earQ*minBW) + np.log(lowFreq + earQ*minBW))/N) * (highFreq + earQ*minBW))[::-1]
  #erb_BW = 24.7 * (0.00437*erb + 1)
  #erb_number = 21.4 * np.log(0.00437*erb + 1)
  return erb
  
  erb = erbScale(30, 11025, 70)

# Use to assign nearest erb center frequency to stft frequency bins.
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]  


#Calculates all features in Essentia extractor, plus some that were calculated in the freesound extractor.
def getFeatures(audio_dict, type ='mono', equal_loudness = False):
  _features = OrderedDict()
  
  
  
  # Exctract most of the low level, tonal, and rhythm features
  for name in gNameTracks:  
      
      extractor = Extractor(dynamics = True, lowLevel = True, midLevel = True,
                            rhythm = True, tuning = True)
      crest = Crest()
      entropy = Entropy()
      spectrum = Spectrum()
      equalLoudness = EqualLoudness()
      flatnessDB = FlatnessDB()
      distributionshape = DistributionShape()
      centralmoments = CentralMoments()
      gfcc = GFCC()
      powerSpectrum = PowerSpectrum()
      melbands = MelBands()
      erbbands = ERBBands()
      w = Windowing(type = 'hann') 
      monoMixer = MonoMixer()
      
      #Dynamics:
      dynamicComplexity = DynamicComplexity()
      replayGain = ReplayGain()
      larm = Larm()
      leq = Leq()
      ldns = Loudness()
      ldnsVickers = LoudnessVickers()
      rms = RMS()
      
      _audio = audio_dict[name]  
      
      if equal_loudness == True:
        
          _audio = equalLoudness(_audio) 
          _audio_eq = _audio
          
      
      #_left, _right = stereoDemuxer(_audio)  
      if 'stereo' in type:
          _CD = getCochleagramDifference(_audio)
          _SPS = getStereoPanningSpectrum(_audio)
          _audio = monoMixer(_audio,2)
         # print 'It s Stereo'              
      
          _features[name] = extractor(_audio)      
          _features[name].merge(_CD)
          _features[name].merge(_SPS)
      
      elif 'mono' in type:
        
          _features[name] = extractor(_audio)    
      
      
      # Extract the rest, such as hpcp crest and entropy.
      
      for i in range(len(_features[name]['tonal.hpcp'])):
          
          _features[name].add('tonal.hpcp_crest', crest(_features[name]['tonal.hpcp'][i]))
          _features[name].add('tonal.hpcp_entropy', entropy(_features[name]['tonal.hpcp'][i]))
      
      
      # Extracts spectral entropy for raw and stem audio.
      # Melbands and ERBBands.
       # Raw:
      
      if equal_loudness == False:
        
          _audio_eq = equalLoudness(_audio)  
          
      crest = Crest()
      entropy = Entropy()
      spectrum = Spectrum()
      w = Windowing(type = 'hann')

      _a = []
      _mel = []
      _erb = []
      
      
      for frame in FrameGenerator(_audio_eq, frameSize=kN, hopSize=kN//2):
          a = spectrum(w(frame))
          #_a.append(a)
          #_mel.append(melbands(a))
          #_erb.append(erbbands(a))
          
             
          _features[name].add('lowLevel.spectral_entropy', entropy(a))  
          
          _features[name].add('lowLevel.melbands', melbands(a))
          #_features[name].add('lowLevel.erbbands', erbbands(a))          
          #_features[name].add('lowLevel.spectrum', a)
      
      
      #Compute perceptual feature: Specific loudness: Total, Relative[], Sharpness and Spread
      g = []
      z = range(1,len(_features[name]['lowLevel.barkbands'][0])+1,1)
      for i in z:
        if i < 15:
          g.append(1)
        else:
          g.append(0.066*np.exp(0.171*i))
      g = np.asarray(g)      
      
      for i in range(len(_features[name]['lowLevel.barkbands'])):

          bark = _features[name]['lowLevel.barkbands'][i]
          specific_loudness = np.power(bark, 0.23)
          total_loudness = np.sum(specific_loudness)
          relative_specific_loudness = (specific_loudness / total_loudness)
          A = 0.11*(np.sum((specific_loudness * g * z)) / total_loudness)
          ET = np.power((total_loudness - np.max(specific_loudness))/total_loudness,2)
          _features[name].add('lowLevel.total_loudness', total_loudness)
          #_features[name].add('lowLevel.relative_specific_loudness', relative_specific_loudness)
          _features[name].add('lowLevel.specific_sharpness', A)
          _features[name].add('lowLevel.specific_spread', ET)

            
      
      # Calculates Melbands: crest and flatnessDB, spread, skewness and kurtosis
      crest = Crest()
      flatnessDB = FlatnessDB()     
      for i in range(len(_features[name]['lowLevel.melbands'])):
          
          _features[name].add('lowLevel.melbands_crest', crest(_features[name]['lowLevel.melbands'][i]))
          _features[name].add('lowLevel.melbands_flatness_db', flatnessDB(_features[name]['lowLevel.melbands'][i]))
          
          _cm = centralmoments(_features[name]['lowLevel.melbands'][i])
          spread, skewness, kurtosis = distributionshape(_cm)
          _features[name].add('lowLevel.melbands_spread', spread)
          _features[name].add('lowLevel.melbands_skewness', skewness)
          _features[name].add('lowLevel.melbands_kurtosis', kurtosis)
      
      # Calculates ERBbands: crest and flatnessDB, spread, skewness and kurtosis
#      crest = Crest()
#      flatnessDB = FlatnessDB() 
#      distributionshape = DistributionShape()
#      centralmoments = CentralMoments()
#      for i in range(len(_features[name]['lowLevel.erbbands'])):
#          
#          _features[name].add('lowLevel.erbbands_crest', crest(_features[name]['lowLevel.erbbands'][i]))
#          _features[name].add('lowLevel.erbbands_flatness_db', flatnessDB(_features[name]['lowLevel.erbbands'][i]))
#          
#          _cm = centralmoments(_features[name]['lowLevel.erbbands'][i])
#          spread, skewness, kurtosis = distributionshape(_cm)
#          _features[name].add('lowLevel.erbbands_spread', spread)
#          _features[name].add('lowLevel.erbbands_skewness', skewness)
#          _features[name].add('lowLevel.erbbands_kurtosis', kurtosis)      
      
      
      
      
      
#      # Extracts GFCC coeffs for raw and stem audio.
#        # Raw:            
#      w = Windowing(type = 'hann')
#      
#      for frame in FrameGenerator(_audio, frameSize=kN, hopSize=kN//2):
#          a = powerSpectrum(w(frame))
#          bands, coeffs = gfcc(a)
#          
#          _features[name].add('lowLevel.gfcc', coeffs) 
#          _features[name].add('lowLevel.gfccbands', bands) 
      
      
      #Compute dynamic features: Dynamic Complexity, LARM, Leq, Level, Loudness, LoudnessEBUR128, LoudnessVickers, Replay gain RMS  
      
      #Dynamic Complexity each 2 seconds, 1 second overlap.
      
      w = Windowing(type = 'hann')
      for frame in FrameGenerator(_audio, frameSize=88200, hopSize=44100, startFromZero = True):
          _dc, loudness = dynamicComplexity(frame)
          _features[name].add('dynamic.dynamic_complexity', _dc) 
          _features[name].add('dynamic.dynamic_complexity_loudness', loudness)
#      _dc, loudness = dynamicComplexity(_audio)
#      _features[name].add('dynamic.dynamic_complexity2', _dc) 
#      _features[name].add('dynamic.dynamic_complexity_loudness2', loudness)    
          
      
      #_rg = replayGain(_audio)
      
      
      #_features[name].add('dynamic.replay_gain', _rg)
      
      for frame in FrameGenerator(_audio, frameSize=4096, hopSize=2048):
          _larm = larm(frame)  
          _leq = leq(frame)
          _loudness = ldns(frame)
          _ldnsVickers = ldnsVickers(frame)
          _rms = rms(frame)
          _features[name].add('dynamic.larm', _larm) 
          _features[name].add('dynamic.leq', _leq) 
          _features[name].add('dynamic.loudness', _loudness) 
          #TODO missing step. 
          _features[name].add('dynamic.loudness_vickers', _ldnsVickers) 
          _features[name].add('dynamic.rms', _rms) 
        
        
      #Compute Temporal Features:
      #Log-attack-time, derivatives envelope, temporal centroid (tctotoal)
      envelope = Envelope()
      logAttackTime = LogAttackTime()
      derivativeSFX = DerivativeSFX()
      tTCToTotal = TCToTotal()
      effectiveDuration = EffectiveDuration()
      strongDecay = StrongDecay()
      flatnessSFX = FlatnessSFX()
      maxToTotal = MaxToTotal()
      minToTotal = MinToTotal()
      #start, stop = StartStopSilence()
      
      env = envelope(_audio)
      
      logAT, startA, stopA = logAttackTime(env)
      derAvAfterMax , maxDerBeforeMax = derivativeSFX(env)
      tc = tTCToTotal(env)
      effDuration = effectiveDuration(env)
      strongD = strongDecay(_audio)
      fSFX = flatnessSFX(env)
      maxtt = maxToTotal(env) 
      mintt = minToTotal(env)
      
      
      _features[name].set('temporal.log_attack_time', logAT)
      _features[name].set('temporal.derAvAfterMax', derAvAfterMax)
      _features[name].set('temporal.maxDerBeforeMax',  maxDerBeforeMax)
      _features[name].set('temporal.tct_to_total',  tc)
      _features[name].set('temporal.effective_duration', effDuration)
      _features[name].set('temporal.strong_decay',  strongD)
      _features[name].set('temporal.flatness_sfx', fSFX)
      _features[name].set('temporal.max_to_total', maxtt)
      _features[name].set('temporal.min_to_total', mintt)
      
      
      
  #DELETED NOT WANTED FEATURES
 
      _features[name].remove('rhythm.bpm')
      _features[name].remove('rhythm.confidence')
      _features[name].remove('tonal.chords_changes_rate')
      _features[name].remove('tonal.chords_number_rate')
      _features[name].remove('tonal.key_strength')
      _features[name].remove('tonal.tuning_diatonic_strength')
      _features[name].remove('tonal.tuning_equal_tempered_deviation')
      _features[name].remove('tonal.tuning_frequency')
      _features[name].remove('tonal.tuning_nontempered_energy_ratio')
      _features[name].remove('tonal.chords_strength')
      _features[name].remove('rhythm.beats_position')
      _features[name].remove('rhythm.bpm_estimates')
      _features[name].remove('rhythm.bpm_intervals')
      _features[name].remove('rhythm.onset_times')
      _features[name].remove('tonal.thpcp')
      _features[name].remove('rhythm.histogram')
      _features[name].remove('tonal.chords_key')
      _features[name].remove('tonal.chords_scale')
      _features[name].remove('tonal.key_key')
      _features[name].remove('tonal.key_scale')
      _features[name].remove('tonal.chords_histogram')
      _features[name].remove('tonal.chords_progression')   
      _features[name].remove('lowLevel.melbands')
      
  
  return _features 
  
 
def getStatisticalFeatures(_dict_features, type = 'weighted'):
  _features = OrderedDict()
  
  exception = {#'temporal.derAvAfterMax' :['mean'],  
#                                            'temporal.log_attack_time' :['mean'],
#                                            'temporal.maxDerBeforeMax' :['mean'],
#                                            'temporal.tct_to_total' :['mean'],
#                                            'temporal.effective_duration' :['mean'],
#                                            'temporal.strong_decay' :['mean'],
#                                            'temporal.flatness_sfx' :['mean'],
#                                            'temporal.max_to_total' :['mean'],
#                                            'temporal.min_to_total' :['mean'],

                                            'lowLevel.barkbands' :['min', 'max', 'median', 'mean',
                                            'var', 'skew', 'dmean',
                                            'dvar', 'dmean2', 'dvar2']}

  aggrPool = PoolAggregator(defaultStats = ['min', 'max', 'median', 'mean',
                                            'var', 'skew', 'kurt', 'dmean',
                                            'dvar', 'dmean2', 'dvar2'], exceptions = exception)
                                            
  for name, pool in _dict_features.items():
    
      _features[name] =  aggrPool(_dict_features[name])  




      if type == 'weighted':
        
          w = _dict_features[name]['lowLevel.total_loudness']
          for feature in _dict_features[name].descriptorNames():
    
              f = _dict_features[name][feature]
              
              if isinstance(f, np.ndarray):   
                
    # Checks if feature is an array, if the length is 432, it weights the mean, var and dev with respect total_loudness
                  
                  if len(f.shape) > 1 and f.shape[0] == 432 : # TODO THIS IS HARDCODED VERY BAD !
                      
                      mean = (np.sum(w*f.T,1)/np.sum(w))  
                      var = np.sum(w*np.power(f-mean,2).T,1)/np.sum(w)
                      _features[name].remove(feature+'.mean')
                      _features[name].remove(feature+'.var')
                      _features[name].set(feature+'.mean', mean)
                      _features[name].set(feature+'.var', var)
    
                  elif len(f.shape) == 1 and f.shape[0] == 432: 
                    
                      mean = np.sum(w*f)/np.sum(w)
                      var = np.sum(w*np.power(f-mean,2))/np.sum(w)
                      _features[name].remove(feature+'.mean')
                      _features[name].remove(feature+'.var')                  
                      _features[name].set(feature+'.mean', mean)
                      _features[name].set(feature+'.var', var)
                  
                  
                  
#Calculates standard deviation:                  
      for feature in _features[name].descriptorNames():
          
          if '.var' in feature:
              _name_feature = feature.split('.var')[0]
              stdev = np.sqrt(_features[name][feature])
              _features[name].set(_name_feature+'.stdev', stdev)

#Applies temporal modeling weightening with total_loudness
                                      

    
  
  return _features  
    

#%%    
# Definition of functions. 
   

  

# Aux function to sort out array so each component is added as a separate feature.
def sortArray(array, name):
  
  pool = essentia.Pool()
  
  if len(array) == 1:
      array = array.T
        
  for idx, value in enumerate(array):
      #if not np.isnan(value):
      #Remove first coefficient mfcc
      if 'mfcc' in name:
          if idx > 0: 
              pool.add(name+'.'+str(idx), value)
      else:  
          pool.add(name+'.'+str(idx), value)
     #pool.add(name+'.'+str(idx), value)
  return pool
        
        
 

def getNameFeaturesIdx(idx):
  a = []
  print '%d most relevant features: \n' % len(idx)
  for i in idx:
      _a  = '.'.join(gNameFeatures[i].split('.')[3::])
      print _a
      a.append(_a)
  
  return a
  
def getNameFeatures():

  a = []
  max_idx = len(gTotalListFeatures) // len (gListTracks)
  features = gTotalListFeatures[0:max_idx]
  
  for n in features:
      f = '.'.join(n.split('.')[3::])
      a.append(f)
      
  return a
  
#change nan for mean of column, and inf for biggest number. 
def getRidNanInf(array):
  
  x = np.copy(array)
  
  found = False  
  
  while found == False:
  
      for i, j in enumerate(np.argmax(x,1)):
        
          if np.isnan(x[i,j]):
              
              x[i,j] = np.nanmean(x[:,j])
              #print 'nan'
              found = True            
      
      if found == False:
          
          #print 'break'
          break
      
      else:
          getRidNanInf(x)
          #print 'repeat'       

  return np.nan_to_num(x)  
  
# return dicts of unwrapped features from the pool of stats
def getDictsOfPoolFeatures(poolStats, type = 'stem'):
  #_pool = pool of feature stats ordered by name of track
  
  _dict = OrderedDict()    
  
  i = 0
  for track in gNameTracks:
      
      pool = essentia.Pool()
      for name in poolStats[track].descriptorNames():
          
          #If a feature is an array it gets unwrapped.
        
        if isinstance(poolStats[track][name], np.ndarray):          
          
          _pool = sortArray(poolStats[track][name], name)
          
          pool.merge(_pool)
        
        
        else:
          
          pool.add(name,poolStats[track][name])
        
        _dict[track] = pool        
        
        
        if i % 770 == 0:
                
                print '%s %s # %d track info added' % (type, kInstrument, i/770)
        i += 1
  
  return _dict

#Return numpy arrays from dict of pool features  
def getArraysFeatures(dictPoolFeatures, type = 'raw'):
  
  x = []
  y = []
  
  for track in dictPoolFeatures.keys():
      
      for feature in dictPoolFeatures[track].descriptorNames():
          
          x.append(dictPoolFeatures[track][feature])
      
      if type == 'raw':
          y.append(0)
      elif type == 'stem':
          y.append(1)    
      
  x = np.asarray(x).reshape(len(gNameTracks), -1)
  
  return getRidNanInf(x), y 
  
#%%
startTime2 = datetime.now()  

#CONSTANT VARIABLES
yamlFile =  './Music/Data/MedleyDB/Audio/info.tracks'   
kInstrument = sys.argv[1]
kSampleRate = 44100
kN = 2048
kType = 'mono'
kFeatures = 1812 #Find a way to not hard-code this. 

if os.path.isfile(yamlFile):        
    yamlInput = YamlInput(filename=yamlFile)
    gPool = yamlInput()
else:
    print "Yaml file not found"


#FEATURE EXTRACTOR
gNameTracks, gRawPath, gStemPath, gStemStereoPath = getInfoTracks(type = 'original')

del gPool
gNameTracksOriginal = gNameTracks[:]

gNameTracksTotal = gRawPath.keys()

#Splits list raw path, so memory does not run out. 
kSplit = 9
k = int(np.ceil(len(gNameTracksTotal)/kSplit))


for i in range(kSplit+k):
    
    startTime = datetime.now()
    
    
    
    gNameTracks = gNameTracksTotal[i*k:(1+i)*k][:]
    
    if len(gNameTracks) > 0:
        
        
        
        gRawAudio, gStemAudio = load_audio(type = kType)  
        
        gStemFeatures = getFeatures(gStemAudio, type = kType, equal_loudness = True)
        
        del gStemAudio 
        gc.collect()
        gRawFeatures = getFeatures(gRawAudio, type = kType, equal_loudness = True)  
        del gRawAudio
        gc.collect()
        gStemFeaturesStats = getStatisticalFeatures(gStemFeatures, type = 'not weighted')
        del gStemFeatures
        gc.collect()        
        gRawFeaturesStats = getStatisticalFeatures(gRawFeatures, type = 'not weighted')
        del gRawFeatures
        gc.collect()
        #saveFeatureStats()
        #
        #
        #del gNameTracks
        #del gRawPath
        #del gStemPath
        #del gStemStereoPath
        #del gRawAudio
        #del gStemAudio
        #del gRawFeatures
        #del gStemFeatures
        #del gStemFeaturesStats
        #del gRawFeaturesStats
        
        
        print '\nExecuted in: %s. \n %d stems/raw %s tracks with %d features each were extracted.\n' % (str(datetime.now() - startTime),
                                                                                                      len(gNameTracks)*2, kInstrument,
                                                                                                      len(gRawFeaturesStats[gNameTracks[0]].descriptorNames()))
        
        
        # OLD WAY:
        #gInputPool = pool_features
        #gPool = getInfoTracks()
        ##del gInputPool
        #startTime = datetime.now()
        #gc.collect() 
        #gTotalListFeatures = gPool.descriptorNames()
        #gDictX, gNameFeatures = getDictFeaturesBass()
        ##del gPool
        #gc.collect() 
        #gListTracks = gDictX.keys()
        ##gNameFeatures = getNameFeatures() 
        ##gReducedFeatures = gNameFeatures[:]
        #X,Y = getArrayFeatures(gDictX) 
        
        #ORGANIZE FEATURES INTO NUMPY ARRAYS
        
        startTime = datetime.now()      
        
        gStem = getDictsOfPoolFeatures(gStemFeaturesStats, type = 'stem')
        del gStemFeaturesStats
        gc.collect()
        
        gRaw = getDictsOfPoolFeatures(gRawFeaturesStats, type = 'raw')
        del gRawFeaturesStats
        gc.collect()
        gTotalListFeatures = gStem[gNameTracks[0]].descriptorNames()
        X_stem, y_stem = getArraysFeatures(gStem, type = 'stem')
        del gStem    
        gc.collect()
        X_raw, y_raw = getArraysFeatures(gRaw, type = 'raw')
        del gRaw
        gc.collect()
        
        if i == 0:
          
            XStem = np.copy(X_stem)
            yStem = np.copy(y_stem)
            XRaw = np.copy(X_raw)
            yRaw = np.copy(y_raw)
        
        else:
            Xs = np.copy(XStem)
            ys = np.copy(yStem)
            Xr = np.copy(XRaw)
            yr = np.copy(yRaw)
            del XStem 
            gc.collect()
            del XRaw
            gc.collect()
            del yStem
            gc.collect()
            del yRaw
            gc.collect()
            XStem = np.append(Xs, X_stem, axis = 0)
            yStem = np.append(ys, y_stem, axis = 0)
            XRaw = np.append(Xr, X_raw, axis = 0)
            yRaw = np.append(yr, y_raw, axis = 0)
            del Xs
            gc.collect()
            del ys
            gc.collect()
            del Xr
            gc.collect()
            del yr
            gc.collect()
        del X_stem
        gc.collect()
        del y_stem
        gc.collect()
        del X_raw
        gc.collect()
        del y_raw
        gc.collect()
        print '\nExecuted in: %s. \n %d stems/raw %s tracks with %d features each were organized.\n' % (str(datetime.now() - startTime),
                                                                                                      len(gNameTracks)*2, kInstrument,
                                                                                                      len(gTotalListFeatures))
    
        
        
        print '\n %d Batch - Executed in: %s. \n ' % (i, str(datetime.now() - startTime))
         
        #del X
        #del Y
        #del gListTracks
        #del gNameFeatures
        ##del gReducedFeatures
        #del gTotalListFeatures
#%%
path = './Music/Data/MedleyDB/Features/%s/%s/' % (sys.argv[2], kType)
if not os.path.exists(path):
  os.makedirs(path)    
np.save(path + kInstrument + '_XRaw.npy', XRaw)
np.save(path + kInstrument + '_yRaw.npy', yRaw)
np.save(path + kInstrument + '_XStem.npy', XStem)
np.save(path + kInstrument + '_yStem.npy', yStem)
np.save(path + kInstrument + '_gNameFeatures.npy', gTotalListFeatures)
np.save(path + kInstrument + '_gListTracks.npy', gNameTracksTotal)
print '\n Total Executed in: %s. \n %d stems/raw %s tracks with %d features each.\n' % (str(datetime.now() - startTime2),
                                                                                                      len(gNameTracksTotal)*2, kInstrument,
                                                                                                      len(gTotalListFeatures))



