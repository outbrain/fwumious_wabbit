

class HelloWorld {
    static {
        // This actually loads the shared object that we'll be creating.
        // The actual location of the .so or .dll may differ based on your
        // platform.
        System.loadLibrary("fw");
    }

    // The rest is just regular ol' Java!
    public static void main(String[] args) {
        String output = HelloWorld.hello("josh");
        System.out.println(output);
    }
}
