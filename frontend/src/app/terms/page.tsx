import LandingNav from "~/components/landing-nav";

export default function Terms() {
  return (
    <div className="w-full">
      <LandingNav />
      <div className="max-w-3xl mx-auto py-12 px-4">

        <h1 className="text-3xl font-bold mb-8">Terms & Conditions</h1>

        <section className="space-y-6">
        <div>
          <h2 className="text-xl font-semibold mb-3">1. Subscription Terms</h2>
          <p>
            Your subscription is a one-time purchase that provides a fixed number of credits. You will have lifetime access
            until your credits are used. Subscriptions do not auto-renew.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-3">2. Payment Terms</h2>
          <p>
            Payments are processed securely through our payment provider.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-3">3. Usage Rights</h2>
          <p>
            This service is for personal use only. 
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-3">4. Refund Policy</h2>
          <p>
            Refunds are evaluated on a case-by-case basis. To request a refund, contact support at <br />
            <a className="underline" href="mailto:priyanshuthapliyal2005@gmail.com">priyanshuthapliyal2005@gmail.com</a> within
            3 days of purchase.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-3">
            5. Service Availability
          </h2>
          <p>
            We strive for 99.9% uptime but cannot guarantee uninterrupted service. We reserve the right to modify or
            discontinue features.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold mb-3">
            6. Limitation of Liability
          </h2>
          <p>
            Our service is provided &quot;as is&quot; without warranties. We are not liable for any damages arising from the use of
            the service to the fullest extent permitted by applicable law.
          </p>
        </div>
        </section>

        <p className="mt-8 text-sm text-gray-500">
          Last updated: {new Date().toLocaleDateString()}
        </p>
      </div>
    </div>
  );
}