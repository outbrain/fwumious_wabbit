package com.examples;

import com.outbrain.fw.FWSession;
import com.outbrain.fw.FWPort;

class Main {
    public static void main(String[] args) {
        System.out.println("Loading fw library");
        try {
            System.loadLibrary("fw");
        } catch (UnsatisfiedLinkError e) {
            System.out.println("Can not load library");
            throw e;
        }
        System.out.println("Library loaded");
        System.out.println("Creating FWSession object");
        FWSession fws = new FWSession("--namespaces AB --keep A --keep B --noconstant");
        System.out.println("Creating FWSPort object");
        FWPort p = new FWPort(fws);
        
        System.out.println("1 learn & prediction " + p.learn(fws, "1 |A a |B b\n"));
        System.out.println("2 learn & prediction "+ p.learn(fws, "1 |A a |B b\n"));
        System.out.println("3 Just predict "+ p.predict(fws, "1 |A a |B b\n"));
        System.out.println("3 Just predict "+ p.predict(fws, "1 |A a |B b\n"));
        System.out.println("Done!");
    }
    
}
