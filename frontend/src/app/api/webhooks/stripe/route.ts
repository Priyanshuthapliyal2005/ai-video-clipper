import { NextResponse } from "next/server";
import Stripe from "stripe";
import { env } from "~/env";
import { db } from "~/server/db";

const stripe = new Stripe(env.STRIPE_SECRET_KEY, {
  apiVersion: "2025-08-27.basil",
});

const webhookSecret = env.STRIPE_WEBHOOK_SECRET;

export async function POST(req: Request) {
  try {
    const body = await req.text();
    const signature = req.headers.get("stripe-signature") ?? "";

    let event: Stripe.Event;

    try {
      event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
    } catch (error) {
      console.error("Webhook signature verification failed:", error);
      return new NextResponse("Webhook signature verification failed", {
        status: 400,
      });
    }

    if (event.type === "checkout.session.completed") {
      const session = event.data.object;
      const customerId = session.customer as string;

      // Only process if payment is actually paid
      if (session.payment_status !== "paid") {
        return new NextResponse(null, { status: 200 });
      }

      let retreivedSession;
      let lineItems;
      
      try {
        retreivedSession = await stripe.checkout.sessions.retrieve(
          session.id,
          { expand: ["line_items", "customer"] },
        );
        lineItems = retreivedSession.line_items;
      } catch (stripeError) {
        console.error("Failed to retrieve session from Stripe:", stripeError);
        return new NextResponse("Session not found", { status: 404 });
      }
      
      if (lineItems && lineItems.data.length > 0) {
        const priceId = lineItems.data[0]?.price?.id ?? undefined;

        if (priceId) {
          let creditsToAdd = 0;

          if (priceId === env.STRIPE_SMALL_CREDIT_PACK) {
            creditsToAdd = 50;
          } else if (priceId === env.STRIPE_MEDIUM_CREDIT_PACK) {
            creditsToAdd = 150;
          } else if (priceId === env.STRIPE_LARGE_CREDIT_PACK) {
            creditsToAdd = 500;
          }

          if (creditsToAdd > 0) {
            try {
              // First, check if user exists
              const existingUser = await db.user.findUnique({
                where: { stripeCustomerId: customerId }
              });
              
              if (!existingUser) {
                // Try to find user by customer email if available
                const customerData = retreivedSession.customer;
                if (customerData && typeof customerData === 'object' && 'email' in customerData) {
                  const customerEmail = customerData.email as string;
                  
                  const userByEmail = await db.user.findUnique({
                    where: { email: customerEmail }
                  });
                  
                  if (userByEmail) {
                    // Update the user with the stripe customer ID
                    await db.user.update({
                      where: { email: customerEmail },
                      data: { stripeCustomerId: customerId }
                    });
                    
                    // Add credits to the user
                    await db.user.update({
                      where: { email: customerEmail },
                      data: {
                        credits: {
                          increment: creditsToAdd,
                        },
                      },
                    });
                  }
                }
              } else {
                // User found, add credits
                await db.user.update({
                  where: { stripeCustomerId: customerId },
                  data: {
                    credits: {
                      increment: creditsToAdd,
                    },
                  },
                });
              }
            } catch (dbError) {
              console.error("Database error:", dbError);
              // Don't return error to Stripe, log it for investigation
            }
          }
        }
      }
    }

    return new NextResponse(null, { status: 200 });
  } catch (error) {
    console.error("Error processing webhook:", error);
    return new NextResponse("Webhook error", { status: 500 });
  }
}